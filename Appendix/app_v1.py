#!/usr/bin/env python3
"""
US Flight Budget Chatbot – v1 UI Upgrade

Enhancements over app.py:
- Smart prompt chips for faster interactions
- Light/Dark theme toggle with persistent preference
- Colorful, accessible, soothing UI
- Background flight GIF support (optional, via /static/flight_bg.gif)
- Browser-based speech-to-text (audio → text) and text-to-speech (text → audio)

Back end logic (slot filling + Amadeus API) mirrors app.py.

Env vars required before running:
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
from flask import Flask, render_template_string, request, session, jsonify
try:
    from dotenv import load_dotenv, find_dotenv
    # Attempt to load from CWD and from the script directory for robustness
    _found = find_dotenv(usecwd=True)
    if _found:
        load_dotenv(_found, override=False)
    _here_env = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(_here_env):
        load_dotenv(_here_env, override=False)
except Exception:
    # dotenv not installed or other issue; environment variables must be set via shell
    pass

# ----------------------------------------------------------------------
# AMADEUS CONFIG
# ----------------------------------------------------------------------

# Note: We will read these from os.environ at call time to reflect .env changes.
AMADEUS_CLIENT_ID = os.environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET")

AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

_amadeus_token: Optional[str] = None
_amadeus_token_expiry: Optional[datetime] = None

# Allow tuning number of Amadeus calls per search window to avoid 429s
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
    # TTL 10 minutes
    if time.time() - ts > 600:
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
        if resp.status_code == 429 or resp.status_code == 503:
            if attempt == max_attempts:
                raise RateLimitedError(f"Amadeus rate limit: {resp.status_code}")
            # Use Retry-After header if present
            retry_after = resp.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else backoff
            except Exception:
                wait = backoff
            time.sleep(min(wait, 8.0))
            backoff = min(backoff * 2, 8.0)
            continue
        if resp.status_code in (401, 403):
            # Upstream handler in call_flight_api will refresh token
            return resp
        resp.raise_for_status()
        return resp
    # Should not reach
    raise RateLimitedError("Amadeus rate limit")

def _get_amadeus_token() -> str:
    global _amadeus_token, _amadeus_token_expiry

    if (
        _amadeus_token is not None
        and _amadeus_token_expiry is not None
        and datetime.utcnow() < _amadeus_token_expiry
    ):
        return _amadeus_token

    # Always read env at call-time so .env changes are respected under reloader
    client_id = os.environ.get("AMADEUS_CLIENT_ID")
    client_secret = os.environ.get("AMADEUS_CLIENT_SECRET")
    if not client_id or not client_secret:
        try:
            from dotenv import load_dotenv as _ld, find_dotenv as _fd
            # Reload from any discoverable .env (cwd) and from script dir, overriding if missing
            _path = _fd(usecwd=True)
            if _path:
                _ld(_path, override=True)
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
    date_pairs = generate_date_pairs(
        date_from, date_to, min_nights, max_nights, max_pairs=MAX_SEARCH_DATE_PAIRS
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
        "- tap a chip below (e.g. Cheaper, Non-stop)\n"
        "- type \"cheaper\" to try a lower budget\n"
        "- type \"fewer stops\" or \"non-stop only\"\n"
        "- type a different month, e.g. \"try July\" or \"move to September\"\n"
        "- or set a new budget directly, e.g. \"set budget to $600\"."
    )

    return "\n".join(lines)


# ----------------------------------------------------------------------
# FLASK APP + ENHANCED CHAT UI
# ----------------------------------------------------------------------

app = Flask(__name__, static_folder="static")
app.secret_key = "replace-with-a-random-secret-string"

HTML_TEMPLATE = r"""
<!doctype html>
<html>
<head>
    <title>US Flight Chatbot — Enhanced</title>
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Cpath fill='%2322c55e' d='M2 38l24-8 10-18 6 4-6 16 18 10-4 4-18-6-8 20-4-2 4-18-20-6z'/%3E%3C/svg%3E" />
    <style>
        :root {
            --bg: #0b1020;         /* dark bg */
            --panel: #0f172a;      /* panel bg */
            --text: #e5e7eb;       /* text on dark */
            --muted: #9ca3af;      /* muted text */
            --accent: #22c55e;     /* green */
            --accent-2: #38bdf8;   /* sky */
            --bubble-bot: #162036; /* bot bubble */
            --bubble-user: #22c55e;/* user bubble */
            --bubble-user-text: #0a0f1d; /* user text on green */
            --border: #1f2937;
            --shadow: rgba(0,0,0,0.5);
        }
        .light {
            --bg: #f8fafc;
            --panel: #ffffff;
            --text: #0b1020;
            --muted: #475569;
            --accent: #16a34a;
            --accent-2: #0284c7;
            --bubble-bot: #eef2ff;
            --bubble-user: #16a34a;
            --bubble-user-text: #ffffff;
            --border: #e2e8f0;
            --shadow: rgba(2,6,23,0.1);
        }
        html, body { height: 100%; }
        body {
            margin: 0;
            font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
            background: var(--bg);
            color: var(--text);
        }
        .bg-wrap {
            position: fixed;
            inset: 0;
            overflow: hidden;
            z-index: -1;
        }
        .bg-wrap::before {
            content: "";
            position: absolute; inset: 0;
            background: radial-gradient(60% 40% at 50% 20%, rgba(56,189,248,0.15), transparent),
                        radial-gradient(40% 30% at 80% 80%, rgba(34,197,94,0.12), transparent);
            filter: blur(40px);
        }
        .bg-wrap .bg-img {
            position: absolute; inset: 0;
            background-image: url('/static/flight_bg.gif');
            background-size: cover;
            background-position: center;
            opacity: 0.15; /* subtle to avoid distraction */
            filter: saturate(1.2) contrast(1.05);
        }
        .container {
            max-width: 980px;
            margin: 24px auto;
            padding: 24px;
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 16px;
            box-shadow: 0 20px 40px var(--shadow);
        }
        .topbar {
            display: flex; align-items: center; justify-content: space-between;
            gap: 12px; margin-bottom: 8px;
        }
        .title { font-weight: 700; letter-spacing: 0.2px; }
        .theme-toggle { display: inline-flex; align-items: center; gap: 8px; }
        .toggle-btn {
            position: relative;
            width: 54px; height: 30px; border-radius: 999px;
            background: var(--border); border: 1px solid var(--border);
            cursor: pointer;
        }
        .toggle-dot {
            position: absolute; top: 50%; left: 2px; transform: translateY(-50%);
            width: 26px; height: 26px; border-radius: 50%;
            background: var(--accent-2);
            transition: left 160ms ease;
        }
        .toggle-on .toggle-dot { left: 26px; background: var(--accent); }

        .helper { color: var(--muted); font-size: 0.9rem; margin-bottom: 8px; }

        .input-row { display: flex; gap: 10px; align-items: center; }
        textarea {
            flex: 1; min-height: 110px;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid var(--border);
            background: transparent;
            color: var(--text);
            resize: vertical;
        }
        .btn {
            display: inline-flex; align-items: center; gap: 8px;
            padding: 10px 14px; border-radius: 999px; border: none;
            font-weight: 600; cursor: pointer; white-space: nowrap;
            background: var(--accent); color: var(--bubble-user-text);
        }
        .btn.secondary { background: var(--accent-2); }
        .btn.ghost { background: transparent; color: var(--text); border: 1px solid var(--border); }
        .btn:disabled { opacity: 0.6; cursor: default; }

        .chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
        .chip { padding: 8px 12px; border-radius: 999px; border: 1px solid var(--border); cursor: pointer; }
        .chip:hover { border-color: var(--accent-2); }

        .chat { margin-top: 18px; max-height: 520px; overflow-y: auto; }
        .bubble {
            padding: 12px 14px; border-radius: 14px; margin-bottom: 10px; max-width: 85%; line-height: 1.45;
        }
        .bubble-user { background: var(--bubble-user); color: var(--bubble-user-text); margin-left: auto; }
        .bubble-bot { background: var(--bubble-bot); color: var(--text); margin-right: auto; }
        .bubble-error { outline: 2px solid #f87171; color: #fecaca; }
        .sysline { color: var(--muted); font-size: 0.85rem; margin: 12px 0 4px; }
        pre { background: transparent; border: 1px dashed var(--border); padding: 8px; border-radius: 8px; color: var(--muted); }
    </style>
</head>
<body>
    <div class="bg-wrap">
        {% if bg_exists %}
        <div class="bg-img"></div>
        {% endif %}
    </div>

    <div class="container">
        <div class="topbar">
            <div class="title">US Flight Budget Chatbot</div>
            <div class="theme-toggle">
                <span id="themeLabel">Dark</span>
                <div id="themeToggle" class="toggle-btn"><div class="toggle-dot"></div></div>
            </div>
        </div>
        <div class="helper">Ask in plain English. Example prompts and chips below.</div>

        <form method="post" id="chatForm">
            <div class="input-row">
                <textarea id="query" name="query" placeholder="Tell me about your US trip...">{{ query or "" }}</textarea>
                <div style="display:flex; flex-direction: column; gap: 8px;">
                    <button type="submit" class="btn" id="sendBtn">Send</button>
                    <button type="button" class="btn secondary" id="speakBtn" title="Speak last response">🔊 Speak</button>
                    <button type="button" class="btn ghost" id="recBtn" title="Voice input">🎙️ Voice</button>
                    <button type="button" class="btn ghost" id="waitingBtn" disabled style="display:none;">⏳ Waiting…</button>
                </div>
            </div>
            <div class="chips" id="smartChips">
                {% if pending_slot == 'origin_code' %}
                    <div class="sysline">Pick an origin:</div>
                    <div class="chip" data-insert="I'm flying from New York City">NYC</div>
                    <div class="chip" data-insert="I'm flying from San Francisco">San Francisco</div>
                    <div class="chip" data-insert="I'm flying from Los Angeles">Los Angeles</div>
                    <div class="chip" data-insert="I'm flying from Chicago">Chicago</div>
                {% elif pending_slot == 'dest_code' %}
                    <div class="sysline">Pick a destination:</div>
                    <div class="chip" data-insert="to San Francisco">San Francisco</div>
                    <div class="chip" data-insert="to New York City">NYC</div>
                    <div class="chip" data-insert="to Miami">Miami</div>
                    <div class="chip" data-insert="to Seattle">Seattle</div>
                {% elif pending_slot == 'year' %}
                    <div class="sysline">Pick a year:</div>
                    <div class="chip" data-insert="in 2025">2025</div>
                    <div class="chip" data-insert="in 2026">2026</div>
                {% elif pending_slot == 'budget_total' %}
                    <div class="sysline">Select a budget:</div>
                    <div class="chip" data-insert="budget $300">$300</div>
                    <div class="chip" data-insert="budget $500">$500</div>
                    <div class="chip" data-insert="budget $800">$800</div>
                {% elif pending_slot == 'trip_length' %}
                    <div class="sysline">Trip length:</div>
                    <div class="chip" data-insert="for 3-5 days">3–5 days</div>
                    <div class="chip" data-insert="for 5-7 days">5–7 days</div>
                    <div class="chip" data-insert="for 7-10 days">7–10 days</div>
                {% elif pending_slot == 'max_stops' %}
                    <div class="sysline">Stops preference:</div>
                    <div class="chip" data-insert="non-stop only">Non-stop</div>
                    <div class="chip" data-insert="up to 1 stop">≤ 1 stop</div>
                    <div class="chip" data-insert="up to 2 stops">≤ 2 stops</div>
                {% else %}
                    <div class="sysline">Quick actions:</div>
                    <div class="chip" data-insert="cheaper">Cheaper</div>
                    <div class="chip" data-insert="fewer stops">Fewer stops</div>
                    <div class="chip" data-insert="non-stop only">Non-stop only</div>
                    <div class="chip" data-insert="try July">Try July</div>
                    <div class="chip" data-insert="set budget to $600">Budget $600</div>
                    <div class="chip" data-insert="reset">Reset</div>
                {% endif %}
            </div>
        </form>

        <div class="chat" id="chat">
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

    <script>
        // Theme persistence
        const root = document.documentElement;
        const body = document.body;
        const toggle = document.getElementById('themeToggle');
        const themeLabel = document.getElementById('themeLabel');
        function setTheme(mode){
            if(mode === 'light') { body.classList.add('light'); toggle.classList.add('toggle-on'); themeLabel.textContent = 'Light'; }
            else { body.classList.remove('light'); toggle.classList.remove('toggle-on'); themeLabel.textContent = 'Dark'; }
            localStorage.setItem('theme', mode);
        }
        const savedTheme = localStorage.getItem('theme') || 'dark';
        setTheme(savedTheme);
        toggle.addEventListener('click', ()=> setTheme(body.classList.contains('light') ? 'dark' : 'light'));

        // Smart chips insert behavior
        document.getElementById('smartChips').addEventListener('click', (e)=>{
            const t = e.target.closest('.chip');
            if(!t) return;
            const insert = t.getAttribute('data-insert') || '';
            const q = document.getElementById('query');
            const spacer = q.value && !q.value.endsWith(' ') ? ' ' : '';
            q.value = (q.value || '') + spacer + insert;
            q.focus();
        });

        // Text-to-Speech (TTS) for last bot message
        document.getElementById('speakBtn').addEventListener('click', ()=>{
            const bubbles = Array.from(document.querySelectorAll('.bubble-bot'));
            if(bubbles.length === 0) return;
            const last = bubbles[bubbles.length - 1];
            const text = last.innerText.replace(/\s+/g, ' ').trim();
            if(window.speechSynthesis){
                const u = new SpeechSynthesisUtterance(text);
                u.rate = 1.0; u.pitch = 1.0; u.lang = 'en-US';
                speechSynthesis.cancel();
                speechSynthesis.speak(u);
            } else {
                alert('Speech Synthesis not supported in this browser.');
            }
        });

        // Speech-to-Text (STT) via Web Speech API
        let recognizing = false;
        let recognition;
        const recBtn = document.getElementById('recBtn');
        function ensureRecognition(){
            const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
            if(!SR) return null;
            const r = new SR();
            r.lang = 'en-US';
            r.interimResults = false;
            r.maxAlternatives = 1;
            return r;
        }
        recBtn.addEventListener('click', ()=>{
            if(recognizing){ recognition.stop(); return; }
            recognition = ensureRecognition();
            if(!recognition){ alert('Speech Recognition not supported in this browser.'); return; }
            recognizing = true; recBtn.textContent = '⏹️ Stop'; recBtn.disabled = false;
            recognition.onresult = (ev)=>{
                const transcript = ev.results[0][0].transcript;
                const q = document.getElementById('query');
                q.value = transcript;
                recognizing = false; recBtn.textContent = '🎙️ Voice';
            };
            recognition.onend = ()=>{ recognizing = false; recBtn.textContent = '🎙️ Voice'; };
            recognition.onerror = ()=>{ recognizing = false; recBtn.textContent = '🎙️ Voice'; };
            recognition.start();
        });

        // Auto-scroll chat to bottom on load
        const chat = document.getElementById('chat');
        chat.scrollTop = chat.scrollHeight;

        // Show waiting state during submit
        const form = document.getElementById('chatForm');
        const sendBtn = document.getElementById('sendBtn');
        const speakBtn = document.getElementById('speakBtn');
        const waitingBtn = document.getElementById('waitingBtn');
        form.addEventListener('submit', ()=>{
            // stop recognition if active
            try { if(recognizing && recognition) recognition.stop(); } catch {}
            sendBtn.disabled = true; sendBtn.textContent = 'Sending…';
            speakBtn.disabled = true; recBtn.disabled = true;
            waitingBtn.style.display = 'inline-flex';
        });
    </script>
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
                            if isinstance(e, RateLimitedError):
                                err = (
                                    "Amadeus rate limit reached. Please wait a bit and try again. "
                                    "Tip: adjust your dates or try fewer quick tweaks in a row."
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
        pending_slot=pending_slot,
        bg_exists=os.path.exists(os.path.join(app.static_folder or 'static', 'flight_bg.gif')),
    )


@app.get("/healthz")
def healthz():
    has_id = bool(os.environ.get("AMADEUS_CLIENT_ID"))
    has_secret = bool(os.environ.get("AMADEUS_CLIENT_SECRET"))
    cached = bool(_amadeus_token)
    return jsonify({
        "env": {
            "AMADEUS_CLIENT_ID": has_id,
            "AMADEUS_CLIENT_SECRET": has_secret,
        },
        "token_cached": cached,
    })


if __name__ == "__main__":
    app.run(debug=True)
