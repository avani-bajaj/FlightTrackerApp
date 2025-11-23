#!/usr/bin/env python3
"""
US Flight Budget Chatbot – Amadeus Flight Offers Search + Slot-Filling Dialog

Requires env vars in the shell BEFORE running `python app.py`:

    AMADEUS_CLIENT_ID
    AMADEUS_CLIENT_SECRET
"""

import os
import time
import re
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple

import requests
from flask import Flask, render_template_string, request, session
try:
    from dotenv import load_dotenv, find_dotenv
    _found = find_dotenv(usecwd=True)
    if _found:
        load_dotenv(_found, override=False)
    _here_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(_here_env):
        load_dotenv(_here_env, override=False)
except Exception:
    pass

# ----------------------------------------------------------------------
# AMADEUS CONFIG
# ----------------------------------------------------------------------

AMADEUS_CLIENT_ID = os.environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET")

AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

_amadeus_token: Optional[str] = None
_amadeus_token_expiry: Optional[datetime] = None

# Reduce API load per search; can be overridden via env
MAX_SEARCH_DATE_PAIRS = int(os.environ.get("MAX_SEARCH_DATE_PAIRS", "5"))

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
    window_mode: str  # "month" or "wide"


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
        "window_mode": None,  # "month" or "wide"
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
        # if user explicitly gave a month, default to month-focused window
        slots["window_mode"] = slots.get("window_mode") or "month"
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
    if slots["year"] is None:
        return "year"
    if slots["budget_total"] is None:
        return "budget_total"
    if slots["min_days"] is None or slots["max_days"] is None:
        return "trip_length"
    if slots["max_stops"] is None:
        return "max_stops"
    return None


def slots_to_intent(slots: Dict) -> Optional[ParsedIntent]:
    try:
        window_mode = slots.get("window_mode")
        if window_mode not in ("month", "wide"):
            # default: if a specific month is known, focus on that month;
            # otherwise, search wide across the year
            window_mode = "month" if slots.get("month") is not None else "wide"
        return ParsedIntent(
            origin_code=slots["origin_code"],
            dest_code=slots["dest_code"],
            year=int(slots["year"]),
            month=slots["month"],
            min_days=int(slots["min_days"]),
            max_days=int(slots["max_days"]),
            budget_total=float(slots["budget_total"]),
            max_stops=int(slots["max_stops"]),
            window_mode=window_mode,
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


def build_search_window(intent: ParsedIntent) -> Tuple[date, date, int, int, bool]:
    """
    Returns (date_from, date_to, min_days, max_days, wide)

    wide = True → we will sample across the whole year.
    """
    if intent.window_mode == "wide":
        df = date(intent.year, 1, 1)
        dt = date(intent.year, 12, 31)
        wide = True
    else:
        if intent.month:
            df, dt = month_range(intent.year, intent.month)
        else:
            # if no explicit month but "month" mode, treat as early-summer window
            df = date(intent.year, 6, 1)
            dt = date(intent.year, 7, 31)
        wide = False
    return df, dt, intent.min_days, intent.max_days, wide


def generate_date_pairs(
    date_from: date,
    date_to: date,
    min_nights: int,
    max_nights: int,
    max_pairs: int = 9,
    wide: bool = False,
) -> List[Tuple[date, date]]:
    pairs: List[Tuple[date, date]] = []

    durations = sorted({min_nights, max_nights, (min_nights + max_nights) // 2})

    if wide:
        # Sample several months across the whole year to keep API calls bounded.
        year = date_from.year
        sample_months = [2, 5, 8, 11]  # roughly once per quarter
        for m in sample_months:
            # use a mid-month departure; clamp to valid range
            dep_day = 10
            dep = date(year, m, dep_day)
            for stay in durations:
                ret = dep + timedelta(days=stay)
                if ret.year == year and ret <= date(year, 12, 31):
                    pairs.append((dep, ret))
                if len(pairs) >= max_pairs:
                    return pairs
        return pairs

    # month-focused window (original logic with start/mid/end)
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

class RateLimitedError(Exception):
    pass

_flight_cache: Dict[Tuple[str, str, str, str], Tuple[dict, float]] = {}
_flight_cache_order: List[Tuple[str, str, str, str]] = []
_FLIGHT_CACHE_MAX = 30

def _cache_get(key: Tuple[str, str, str, str]) -> Optional[dict]:
    item = _flight_cache.get(key)
    if not item:
        return None
    payload, ts = item
    if time.time() - ts > 600:  # 10 minutes TTL
        try:
            del _flight_cache[key]
            _flight_cache_order.remove(key)
        except Exception:
            pass
        return None
    return payload

def _cache_set(key: Tuple[str, str, str, str], payload: dict) -> None:
    _flight_cache[key] = (payload, time.time())
    try:
        _flight_cache_order.remove(key)
    except ValueError:
        pass
    _flight_cache_order.append(key)
    while len(_flight_cache_order) > _FLIGHT_CACHE_MAX:
        old = _flight_cache_order.pop(0)
        _flight_cache.pop(old, None)

def _http_get_with_retry(url: str, headers: dict, params: dict, max_attempts: int = 3) -> requests.Response:
    backoff = 1.0
    for attempt in range(1, max_attempts + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        if resp.status_code in (429, 503):
            if attempt == max_attempts:
                raise RateLimitedError(f"Amadeus rate limit: {resp.status_code}")
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else backoff
            except Exception:
                wait = backoff
            time.sleep(min(wait, 8.0))
            backoff = min(backoff * 2, 8.0)
            continue
        if resp.status_code in (401, 403):
            return resp
        resp.raise_for_status()
        return resp
    raise RateLimitedError("Amadeus rate limit")
def _get_amadeus_token() -> str:
    global _amadeus_token, _amadeus_token_expiry

    if (
        _amadeus_token is not None
        and _amadeus_token_expiry is not None
        and datetime.utcnow() < _amadeus_token_expiry
    ):
        return _amadeus_token

    # Read env at call time and attempt to reload .env if missing
    client_id = os.environ.get("AMADEUS_CLIENT_ID")
    client_secret = os.environ.get("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        try:
            from dotenv import load_dotenv as _ld, find_dotenv as _fd
            p = _fd(usecwd=True)
            if p:
                _ld(p, override=True)
            _here = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(_here):
                _ld(_here, override=True)
            client_id = os.environ.get("AMADEUS_CLIENT_ID")
            client_secret = os.environ.get("AMADEUS_CLIENT_SECRET")
        except Exception:
            pass
    if not client_id or not client_secret:
        raise RuntimeError("AMADEUS_CLIENT_ID or AMADEUS_CLIENT_SECRET not set.")

    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
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


def _parse_flight_offers_response(payload: dict) -> List[Itinerary]:
    """
    Parse Amadeus response into Itinerary objects.
    No filtering by stops/budget here; caller handles it.
    """
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
    wide: bool,
) -> List[Itinerary]:
    token = _get_amadeus_token()
    headers = {"Authorization": f"Bearer {token}"}
    date_pairs = generate_date_pairs(
        date_from, date_to, min_nights, max_nights, max_pairs=MAX_SEARCH_DATE_PAIRS, wide=wide
    )

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

        key = (
            params["originLocationCode"],
            params["destinationLocationCode"],
            params["departureDate"],
            params["returnDate"],
        )
        cached = _cache_get(key)
        if cached is None:
            resp = _http_get_with_retry(
                AMADEUS_FLIGHT_OFFERS_URL,
                headers=headers,
                params=params,
                max_attempts=3,
            )
            if resp.status_code in (401, 403):
                token = _get_amadeus_token()
                headers["Authorization"] = f"Bearer {token}"
                resp = _http_get_with_retry(
                    AMADEUS_FLIGHT_OFFERS_URL,
                    headers=headers,
                    params=params,
                    max_attempts=3,
                )
            payload = resp.json()
            _cache_set(key, payload)
        else:
            payload = cached
        itins = _parse_flight_offers_response(payload)
        all_itins.extend(itins)

    # Deduplicate (price, dates, stops)
    unique: List[Itinerary] = []
    seen = set()
    for it in all_itins:
        key = (
            round(it.price),
            it.currency,
            it.outbound_date,
            it.inbound_date,
            it.stops_outbound,
            it.stops_inbound,
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(it)

    # Filter by stops and budget
    exact = [
        it
        for it in unique
        if it.stops_outbound == max_stops
        and it.stops_inbound == max_stops
        and it.price <= budget_total
    ]

    if exact:
        candidates = exact
    else:
        candidates = [
            it
            for it in unique
            if it.stops_outbound <= max_stops
            and it.stops_inbound <= max_stops
            and it.price <= budget_total
        ]

    candidates_sorted = sorted(candidates, key=lambda x: x.price)
    return candidates_sorted[:10]


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

    def stop_label(n: int) -> str:
        if n == 0:
            return "non-stop"
        if n == 1:
            return "1 stop"
        return f"{n} stops"

    window_desc = "across the year" if intent.window_mode == "wide" else "in that month"
    lines: List[str] = []
    lines.append(
        f"Here are some options for {intent.origin_code} → {intent.dest_code} "
        f"in {intent.year}, budget around ${intent.budget_total:.0f}, "
        f"{intent.min_days}–{intent.max_days} days, up to {intent.max_stops} stops "
        f"({window_desc})."
    )
    lines.append("")

    for i, it in enumerate(top, start=1):
        label = " (cheapest)" if it is cheapest else ""
        stops_out_str = stop_label(it.stops_outbound)
        stops_in_str = stop_label(it.stops_inbound)
        lines.append(f"Option {i}{label}")
        lines.append(f"  Price:  ~${it.price:.0f} {it.currency}")
        lines.append(f"  Dates:  {it.outbound_date.isoformat()} → {it.inbound_date.isoformat()}")
        lines.append(f"  Stops:  outbound {stops_out_str}, inbound {stops_in_str}")
        lines.append(f"  Time:   ≈{it.total_duration_hours:.1f} hours total")
        lines.append("")

    lines.append("How do you want to tweak this?")
    lines.append('- type "cheaper" to widen the search (more months, more stops, lower budget)')
    lines.append('- type "fewer stops" or "non-stop only" to tighten stops')
    lines.append('- type a different month, e.g. "try July" or "move to September"')
    lines.append('- or set a new budget directly, e.g. "set budget to $600".')

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
            white-space: pre-wrap; /* preserve newlines */
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
                {{ msg.text }}
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
                    # lower budget, widen window, allow more stops (up to 2)
                    slots["budget_total"] = max(50.0, last_intent.budget_total * 0.9)
                    slots["max_stops"] = max(last_intent.max_stops, 2)
                    slots["window_mode"] = "wide"
                elif "non-stop" in text_low or "nonstop" in text_low or "non stop" in text_low:
                    # force non-stop, month-focused, slightly higher budget if user didn't override
                    slots["max_stops"] = 0
                    slots["window_mode"] = "month"
                    if slots.get("budget_total") is None:
                        slots["budget_total"] = last_intent.budget_total * 1.1
                elif "fewer stops" in text_low:
                    new_max = max(0, last_intent.max_stops - 1)
                    slots["max_stops"] = new_max
                    slots["window_mode"] = last_intent.window_mode
                elif "more stops" in text_low:
                    slots["max_stops"] = last_intent.max_stops + 1
                    # more stops → often cheaper; widen if not already wide
                    slots["window_mode"] = "wide"

                # month change (e.g. "try july", "move to september")
                new_month = extract_month(text_low)
                if new_month:
                    slots["month"] = new_month
                    slots["window_mode"] = "month"

                # explicit new budget (overrides cheaper/non-stop budget change)
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
                        df, dt, min_nights, max_nights, wide = build_search_window(intent)
                        try:
                            itins = call_flight_api(
                                origin=intent.origin_code,
                                dest=intent.dest_code,
                                date_from=df,
                                date_to=dt,
                                min_nights=min_nights,
                                max_nights=max_nights,
                                max_stops=intent.max_stops,
                                budget_total=intent.budget_total,
                                wide=wide,
                            )
                            summary = summarize_itineraries(intent, itins)
                            history.append({"role": "bot", "text": summary, "error": False})
                            last_intent = intent
                        except Exception as e:
                            if isinstance(e, RateLimitedError):
                                err = (
                                    "Amadeus rate limit reached. Please wait a bit and try again. "
                                    "Tip: adjust dates or try fewer tweaks in a row."
                                )
                            else:
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
