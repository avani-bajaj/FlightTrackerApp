#!/usr/bin/env python3
"""
US Flight Budget Chatbot – Amadeus Flight Offers Search + Slot-Filling Dialog

- Multi-turn chat:
    * Bot asks follow-up questions until it has:
      origin, destination, year, (optional month), trip length, budget, max stops.
- US routes only (predefined airport list).
- Uses Amadeus Self-Service Flight Offers Search API in TEST environment:
    * OAuth2 client_credentials → access_token
    * GET /v2/shopping/flight-offers

Env vars required in the shell BEFORE running `python app.py`:

    AMADEUS_CLIENT_ID
    AMADEUS_CLIENT_SECRET
"""

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, render_template_string, request, session

# ----------------------------------------------------------------------
# AMADEUS CONFIG
# ----------------------------------------------------------------------

AMADEUS_CLIENT_ID = os.environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET")

AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

# simple in-process token cache
_amadeus_token: Optional[str] = None
_amadeus_token_expiry: Optional[datetime] = None

# ----------------------------------------------------------------------
# DOMAIN CONFIG: US-ONLY AIRPORTS / CITY MAPPING
# ----------------------------------------------------------------------

US_AIRPORTS = {
    "JFK", "EWR", "LGA",          # NYC
    "SFO", "OAK", "SJC",          # Bay Area
    "LAX", "BUR", "SNA", "ONT", "LGB",  # LA
    "ORD", "MDW",                 # Chicago
    "SEA",                        # Seattle
    "BOS",                        # Boston
    "ATL",                        # Atlanta
    "DFW", "DAL",                 # Dallas
    "IAH", "HOU",                 # Houston
    "DEN",                        # Denver
    "PHX",                        # Phoenix
    "LAS",                        # Las Vegas
    "MIA", "FLL", "PBI",          # South Florida
}

CITY_TO_AIRPORT = {
    "new york city": "JFK",
    "nyc": "JFK",
    "new york": "JFK",
    "san francisco": "SFO",
    "los angeles": "LAX",
    "la": "LAX",
    "chicago": "ORD",
    "miami": "MIA",
    "seattle": "SEA",
    "boston": "BOS",
    "atlanta": "ATL",
    "dallas": "DFW",
    "houston": "IAH",
    "denver": "DEN",
    "phoenix": "PHX",
    "las vegas": "LAS",
}

MONTHS = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

SLOT_QUESTIONS = {
    "origin_code": "Which US city are you flying out from?",
    "dest_code": "Which US city are you flying to?",
    "year": "What year are you planning to travel?",
    "budget_total": "What is your total flight budget in USD?",
    "trip_length": "How many days do you want to stay? For example: 4-7 days.",
    "max_stops": "What is the maximum number of stops you are okay with? (0 = non-stop, 1, 2, etc.)",
}

# ----------------------------------------------------------------------
# DATA STRUCTURES
# ----------------------------------------------------------------------

@dataclass
class ParsedIntent:
    origin_code: str
    dest_code: str
    year: int
    month: Optional[int]
    min_days: int
    max_days: int
    budget_total: float
    max_stops: int


@dataclass
class Itinerary:
    price: float
    currency: str
    outbound_date: date
    inbound_date: date
    total_duration_hours: float
    stops_outbound: int
    stops_inbound: int
    summary: str


# ----------------------------------------------------------------------
# NLP HELPERS
# ----------------------------------------------------------------------

def extract_year(text: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


def extract_month(text: str) -> Optional[int]:
    low = text.lower()
    for name, num in MONTHS.items():
        if name in low:
            return num
    return None


def extract_budget(text: str) -> Optional[float]:
    m = re.search(r"\$\s*([0-9]{2,6})", text)
    if m:
        return float(m.group(1))
    m = re.search(
        r"(?:budget|spend|under|less than|up to|maximum(?: of)?)\D*([0-9]{2,6})",
        text,
        flags=re.IGNORECASE,
    )
    return float(m.group(1)) if m else None


def extract_trip_length(text: str) -> Tuple[Optional[int], Optional[int]]:
    low = text.lower()
    m = re.search(r"(\d+)\s*(?:-|to)\s*(\d+)\s*days?", low)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = min(a, b), max(a, b)
        return max(lo, 1), max(hi, lo)
    m = re.search(r"for\s+(\d+)\s*days?", low)
    if m:
        d = int(m.group(1))
        return max(d, 1), max(d, 1)
    return None, None


def extract_max_stops(text: str) -> Optional[int]:
    low = text.lower()
    if "non-stop" in low or "nonstop" in low or "non stop" in low or "direct" in low:
        return 0
    m = re.search(r"(\d+)\s*stops?", low)
    return int(m.group(1)) if m else None


def extract_cities(text: str) -> Tuple[Optional[str], Optional[str]]:
    low = text.lower()
    m = re.search(r"from (.+?) to (.+?)(?:\.|,| in |$)", low)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    found = [c for c in CITY_TO_AIRPORT.keys() if c in low]
    if len(found) >= 2:
        return found[0], found[1]
    return None, None


def city_to_airport_code(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    return CITY_TO_AIRPORT.get(city.lower().strip())


# ----------------------------------------------------------------------
# SLOT MANAGEMENT
# ----------------------------------------------------------------------

def init_slots() -> Dict:
    return {
        "origin_code": None,
        "dest_code": None,
        "year": None,
        "month": None,
        "min_days": None,
        "max_days": None,
        "budget_total": None,
        "max_stops": None,
    }


def update_slots_from_text(slots: Dict, text: str) -> Dict:
    origin_city, dest_city = extract_cities(text)
    origin_code = city_to_airport_code(origin_city)
    dest_code = city_to_airport_code(dest_city)
    year = extract_year(text)
    month = extract_month(text)
    budget = extract_budget(text)
    min_days, max_days = extract_trip_length(text)
    max_stops = extract_max_stops(text)

    if origin_code:
        slots["origin_code"] = origin_code
    if dest_code:
        slots["dest_code"] = dest_code
    if year:
        slots["year"] = year
    if month:
        slots["month"] = month
    if budget is not None:
        slots["budget_total"] = float(budget)
    if min_days is not None and max_days is not None:
        slots["min_days"] = min_days
        slots["max_days"] = max_days
    if max_stops is not None:
        slots["max_stops"] = max_stops

    # enforce US-only
    if slots["origin_code"] and slots["origin_code"] not in US_AIRPORTS:
        slots["origin_code"] = None
    if slots["dest_code"] and slots["dest_code"] not in US_AIRPORTS:
        slots["dest_code"] = None

    return slots


def next_missing_slot(slots: Dict) -> Optional[str]:
    if slots["origin_code"] is None:
        return "origin_code"
    if slots["dest_code"] is None:
        return "dest_code"
    # year/month become optional; only ask if user volunteered partial info.
    if slots["year"] is None:
        return None
    if slots["budget_total"] is None:
        return "budget_total"
    if slots["min_days"] is None or slots["max_days"] is None:
        return "trip_length"
    if slots["max_stops"] is None:
        return "max_stops"
    return None


def slots_to_intent(slots: Dict) -> Optional[ParsedIntent]:
    try:
        return ParsedIntent(
            origin_code=slots["origin_code"],
            dest_code=slots["dest_code"],
            year=int(slots["year"]),
            month=slots["month"],
            min_days=int(slots["min_days"]),
            max_days=int(slots["max_days"]),
            budget_total=float(slots["budget_total"]),
            max_stops=int(slots["max_stops"]),
        )
    except Exception:
        return None


# ----------------------------------------------------------------------
# DATE WINDOW + DATE PAIRS
# ----------------------------------------------------------------------

def month_range(year: int, month: int) -> Tuple[date, date]:
    first = date(year, month, 1)
    if month == 12:
        last = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        last = date(year, month + 1, 1) - timedelta(days=1)
    return first, last


def build_search_window(intent: ParsedIntent) -> Tuple[date, date, int, int]:
    if intent.month:
        date_from, date_to = month_range(intent.year, intent.month)
    else:
        date_from = date(intent.year, 6, 1)
        date_to = date(intent.year, 7, 31)
    return date_from, date_to, intent.min_days, intent.max_days


def generate_date_pairs(
    date_from: date,
    date_to: date,
    min_nights: int,
    max_nights: int,
    max_pairs: int = 9,
) -> List[Tuple[date, date]]:
    pairs: List[Tuple[date, date]] = []

    window_days = (date_to - date_from).days
    if window_days <= 0:
        stay = min_nights
        ret = date_from + timedelta(days=stay)
        if ret <= date_to:
            return [(date_from, ret)]
        return []

    dep_candidates = []
    dep_candidates.append(date_from)
    mid_offset = max(0, window_days // 2)
    dep_candidates.append(date_from + timedelta(days=mid_offset))
    end_offset = max(0, window_days - max_nights)
    dep_candidates.append(date_from + timedelta(days=end_offset))
    dep_candidates = sorted(set(d for d in dep_candidates if d <= date_to))

    durations = sorted({min_nights, max_nights, (min_nights + max_nights) // 2})

    for dep in dep_candidates:
        for stay in durations:
            ret = dep + timedelta(days=stay)
            if ret <= date_to:
                pairs.append((dep, ret))
            if len(pairs) >= max_pairs:
                return pairs

    return pairs


# ----------------------------------------------------------------------
# AMADEUS TOKEN + FLIGHT OFFERS
# ----------------------------------------------------------------------

def _get_amadeus_token() -> str:
    global _amadeus_token, _amadeus_token_expiry

    if (
        _amadeus_token is not None
        and _amadeus_token_expiry is not None
        and datetime.utcnow() < _amadeus_token_expiry
    ):
        return _amadeus_token

    if not AMADEUS_CLIENT_ID or not AMADEUS_CLIENT_SECRET:
        raise RuntimeError("AMADEUS_CLIENT_ID or AMADEUS_CLIENT_SECRET not set.")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": AMADEUS_CLIENT_ID,
        "client_secret": AMADEUS_CLIENT_SECRET,
    }

    resp = requests.post(AMADEUS_AUTH_URL, headers=headers, data=data, timeout=20)
    resp.raise_for_status()
    payload = resp.json()
    token = payload["access_token"]
    expires_in = int(payload.get("expires_in", 1800))

    _amadeus_token = token
    _amadeus_token_expiry = datetime.utcnow() + timedelta(seconds=expires_in - 60)

    return token


def _parse_iso8601_duration_to_hours(duration: str) -> float:
    if not duration or not duration.startswith("PT"):
        return 0.0
    h = 0
    m = 0
    mh = re.search(r"(\d+)H", duration)
    mm = re.search(r"(\d+)M", duration)
    if mh:
        h = int(mh.group(1))
    if mm:
        m = int(mm.group(1))
    return float(h) + float(m) / 60.0


def _parse_flight_offers_response(
    payload: dict,
    max_stops: int,
) -> List[Itinerary]:
    itineraries: List[Itinerary] = []
    data = payload.get("data", [])
    currency = None

    for item in data:
        price_block = item.get("price", {})
        total_str = price_block.get("total")
        if total_str is None:
            continue
        try:
            price = float(total_str)
        except ValueError:
            continue

        currency = price_block.get("currency", "USD")

        its = item.get("itineraries", [])
        if len(its) < 2:
            continue

        out_it = its[0]
        in_it = its[1]

        out_segments = out_it.get("segments", [])
        in_segments = in_it.get("segments", [])
        if not out_segments or not in_segments:
            continue

        stops_out = max(0, len(out_segments) - 1)
        stops_in = max(0, len(in_segments) - 1)
        if stops_out > max_stops or stops_in > max_stops:
            continue

        out_first_dep = out_segments[0].get("departure", {})
        in_last_arr = in_segments[-1].get("arrival", {})

        out_at = out_first_dep.get("at")
        in_at = in_last_arr.get("at")
        if not out_at or not in_at:
            continue

        out_date_str = out_at.split("T")[0]
        in_date_str = in_at.split("T")[0]

        try:
            out_date = date.fromisoformat(out_date_str)
            in_date = date.fromisoformat(in_date_str)
        except Exception:
            continue

        dur_out = _parse_iso8601_duration_to_hours(out_it.get("duration", ""))
        dur_in = _parse_iso8601_duration_to_hours(in_it.get("duration", ""))
        total_hours = dur_out + dur_in

        summary = (
            f"{price:.0f} {currency}, {out_date.isoformat()} → {in_date.isoformat()}, "
            f"{stops_out} stops out / {stops_in} back, ≈{total_hours:.1f}h"
        )

        itineraries.append(
            Itinerary(
                price=price,
                currency=currency,
                outbound_date=out_date,
                inbound_date=in_date,
                total_duration_hours=total_hours,
                stops_outbound=stops_out,
                stops_inbound=stops_in,
                summary=summary,
            )
        )

    return itineraries


def call_flight_api(
    origin: str,
    dest: str,
    date_from: date,
    date_to: date,
    min_nights: int,
    max_nights: int,
    max_stops: int,
    budget_total: float,
) -> List[Itinerary]:
    token = _get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}
    date_pairs = generate_date_pairs(date_from, date_to, min_nights, max_nights)

    all_itins: List[Itinerary] = []
    for dep_date, ret_date in date_pairs:
        params = {
            "originLocationCode": origin,
            "destinationLocationCode": dest,
            "departureDate": dep_date.isoformat(),
            "returnDate": ret_date.isoformat(),
            "adults": 1,
            "currencyCode": "USD",
            "max": 20,
        }

        resp = requests.get(
            AMADEUS_FLIGHT_OFFERS_URL,
            headers=headers,
            params=params,
            timeout=30,
        )
        if resp.status_code in (401, 403):
            token = _get_amadeus_token()
            headers["Authorization"] = f"Bearer {token}"
            resp = requests.get(
                AMADEUS_FLIGHT_OFFERS_URL,
                headers=headers,
                params=params,
                timeout=30,
            )

        resp.raise_for_status()
        payload = resp.json()
        itins = _parse_flight_offers_response(payload, max_stops=max_stops)

        itins = [it for it in itins if it.price <= budget_total]

        all_itins.extend(itins)

        if len(all_itins) >= 10:
            break

    return all_itins


def summarize_itineraries(intent: ParsedIntent, itineraries: List[Itinerary]) -> str:
    if not itineraries:
        return (
            f"I couldn't find any itineraries for {intent.origin_code} → {intent.dest_code} "
            f"in {intent.year} within about ${intent.budget_total:.0f} "
            f"for a {intent.min_days}–{intent.max_days}-day trip "
            f"and up to {intent.max_stops} stops.\n\n"
            f"You can try one of these tweaks:\n"
            f"- increase budget (e.g. \"set budget to $600\")\n"
            f"- allow more stops (e.g. \"allow 2 stops\")\n"
            f"- move to a different month (e.g. \"try July instead\")."
        )

    itineraries_sorted = sorted(itineraries, key=lambda x: x.price)
    top = itineraries_sorted[:3]
    cheapest = top[0]

    lines: List[str] = []
    lines.append(
        f"Here are some options for {intent.origin_code} → {intent.dest_code} "
        f"in {intent.year}, budget around ${intent.budget_total:.0f}, "
        f"{intent.min_days}–{intent.max_days} days, up to {intent.max_stops} stops:\n"
    )

    for i, it in enumerate(top, start=1):
        label = " (cheapest)" if it is cheapest else ""
        lines.append(
            f"{i}) Price: ~${it.price:.0f} {it.currency}{label}\n"
            f"   Dates:  {it.outbound_date.isoformat()} → {it.inbound_date.isoformat()}\n"
            f"   Stops:  {it.stops_outbound} out / {it.stops_inbound} back\n"
            f"   Time:   ≈{it.total_duration_hours:.1f}h total\n"
        )

    lines.append(
        "How do you want to tweak this?\n"
        "- type \"cheaper\" to try a lower budget\n"
        "- type \"fewer stops\" or \"non-stop only\"\n"
        "- type a different month, e.g. \"try July\" or \"move to September\"\n"
        "- or set a new budget directly, e.g. \"set budget to $600\"."
    )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FLASK APP + CHAT UI
# ----------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "replace-with-a-random-secret-string"

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
    <title>US Flight Budget Chatbot (Amadeus, Slot Filling)</title>
    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            margin: 0;
            background: #0f172a;
            color: #e5e7eb;
        }
        .container {
            max-width: 900px;
            margin: 40px auto;
            padding: 24px;
            background: #020617;
            border-radius: 16px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.5);
        }
        h1 { margin-top: 0; }
        textarea {
            width: 100%;
            height: 120px;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #1f2937;
            background: #020617;
            color: #e5e7eb;
            resize: vertical;
        }
        button {
            margin-top: 12px;
            padding: 10px 18px;
            border-radius: 999px;
            border: none;
            background: #22c55e;
            color: #020617;
            font-weight: 600;
            cursor: pointer;
        }
        button:hover { opacity: 0.9; }
        .chat {
            margin-top: 24px;
            max-height: 500px;
            overflow-y: auto;
        }
        .bubble {
            padding: 10px 14px;
            border-radius: 16px;
            margin-bottom: 10px;
            max-width: 80%;
            line-height: 1.4;
        }
        .bubble-user {
            background: #22c55e;
            color: #020617;
            margin-left: auto;
        }
        .bubble-bot {
            background: #111827;
            color: #e5e7eb;
            margin-right: auto;
        }
        .bubble-error {
            border: 1px solid #f97373;
            color: #fecaca;
        }
        pre {
            background: #020617;
            border-radius: 8px;
            padding: 8px 10px;
            font-size: 0.85rem;
            border: 1px solid #1f2937;
            color: #9ca3af;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>US Flight Budget Chatbot (Amadeus, Slot Filling)</h1>
    <p>Example:</p>
    <pre>I want to travel from New York City to San Francisco in June 2026.
I can't spend more than $400 on flights and want to stay 4-7 days.
I'm okay with up to 2 stops.</pre>

    <form method="post">
        <textarea name="query" placeholder="Tell me about your US trip...">{{ query or "" }}</textarea><br>
        <button type="submit">Send</button>
    </form>

    <div class="chat">
        {% for msg in history %}
            <div class="bubble {% if msg.role == 'user' %}bubble-user{% else %}
                                bubble-bot{% if msg.error %} bubble-error{% endif %}
                                {% endif %}">
                {% if msg.role == 'bot' %}
                    {{ msg.text | e | replace('\\n','<br>') | safe }}
                {% else %}
                    {{ msg.text }}
                {% endif %}
            </div>
        {% endfor %}
    </div>
</div>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    history: List[Dict] = session.get("history", [])
    slots: Dict = session.get("slots") or init_slots()
    pending_slot: Optional[str] = session.get("pending_slot")

    last_intent_raw = session.get("last_intent")
    last_intent: Optional[ParsedIntent] = None
    if last_intent_raw:
        try:
            last_intent = ParsedIntent(**last_intent_raw)
        except Exception:
            last_intent = None

    query = ""

    if request.method == "POST":
        query = request.form.get("query", "").strip()
        if query.lower() in {"reset", "start over"}:
            history = []
            slots = init_slots()
            pending_slot = None
            last_intent = None
        elif query:
            history.append({"role": "user", "text": query, "error": False})

            text_low = query.lower()

            # refinement layer using last search
            if last_intent:
                if "cheaper" in text_low:
                    slots["budget_total"] = max(50.0, last_intent.budget_total * 0.9)

                if "non-stop" in text_low or "nonstop" in text_low or "non stop" in text_low:
                    slots["max_stops"] = 0
                elif "fewer stops" in text_low:
                    slots["max_stops"] = max(0, last_intent.max_stops - 1)
                elif "more stops" in text_low:
                    slots["max_stops"] = last_intent.max_stops + 1

                new_month = extract_month(text_low)
                if new_month:
                    slots["month"] = new_month

                new_budget = extract_budget(query)
                if new_budget is not None:
                    slots["budget_total"] = float(new_budget)

            # normal parsing for any other info
            slots = update_slots_from_text(slots, query)

            if pending_slot == "budget_total" and slots["budget_total"] is None:
                bot_text = "I still didn't catch your budget. Tell me your total flight budget in USD."
                history.append({"role": "bot", "text": bot_text, "error": False})
            elif pending_slot == "trip_length" and (slots["min_days"] is None or slots["max_days"] is None):
                bot_text = "I still need your trip length. For example: 4-7 days or for 5 days."
                history.append({"role": "bot", "text": bot_text, "error": False})
            elif pending_slot == "max_stops" and slots["max_stops"] is None:
                bot_text = "How many stops at most are you okay with? 0 for non-stop, or 1, 2, etc."
                history.append({"role": "bot", "text": bot_text, "error": False})
            else:
                missing = next_missing_slot(slots)
                if missing is not None:
                    pending_slot = missing
                    bot_text = SLOT_QUESTIONS[missing]
                    history.append({"role": "bot", "text": bot_text, "error": False})
                else:
                    pending_slot = None
                    intent = slots_to_intent(slots)
                    if intent is None:
                        err = "Something went wrong while assembling your request. Say 'reset' to start over."
                        history.append({"role": "bot", "text": err, "error": True})
                    else:
                        date_from, date_to, min_nights, max_nights = build_search_window(intent)
                        try:
                            itins = call_flight_api(
                                origin=intent.origin_code,
                                dest=intent.dest_code,
                                date_from=date_from,
                                date_to=date_to,
                                min_nights=min_nights,
                                max_nights=max_nights,
                                max_stops=intent.max_stops,
                                budget_total=intent.budget_total,
                            )
                            summary = summarize_itineraries(intent, itins)
                            history.append({"role": "bot", "text": summary, "error": False})
                            last_intent = intent
                        except Exception as e:
                            err = f"Error calling Amadeus API: {e}"
                            history.append({"role": "bot", "text": err, "error": True})

        session["history"] = history
        session["slots"] = slots
        session["pending_slot"] = pending_slot
        session["last_intent"] = last_intent.__dict__ if last_intent else None

    return render_template_string(
        HTML_TEMPLATE,
        history=history,
        query=query,
    )


if __name__ == "__main__":
    app.run(debug=True)
