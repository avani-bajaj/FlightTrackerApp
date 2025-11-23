#!/usr/bin/env python3
"""
AirScout — Streamlit Chatbot UI (v4)

- Streamlit chat UI with smart prompt chips above and chat at bottom
- Dark/Light toggle (custom CSS variables)
- Subtle background GIF support (reads static/flight_bg.gif if present)
- Browser-based TTS (Speak last reply) + STT (Voice → inserts recognized text)
- Backend logic reused from Flask version: slot-filling + Amadeus API with rate limiting, caching

Run:
  streamlit run app_v4.py

Env:
  AMADEUS_CLIENT_ID, AMADEUS_CLIENT_SECRET (load via .env if present)
  Optional: MAX_SEARCH_DATE_PAIRS
"""

import base64
import html
import os
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from typing import Dict, List, Optional, Tuple

import requests
import streamlit as st

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

# --------------------------- Domain + config ---------------------------

AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"

US_AIRPORTS = {
    "JFK", "EWR", "LGA",
    "SFO", "OAK", "SJC",
    "LAX", "BUR", "SNA", "ONT", "LGB",
    "ORD", "MDW",
    "SEA", "BOS", "ATL",
    "DFW", "DAL", "IAH", "HOU",
    "DEN", "PHX", "LAS",
    "MIA", "FLL", "PBI",
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

MAX_SEARCH_DATE_PAIRS = int(os.environ.get("MAX_SEARCH_DATE_PAIRS", "5"))


# --------------------------- Data structures ---------------------------

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


# --------------------------- NLP helpers ---------------------------

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


# --------------------------- Slot management ---------------------------

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


# --------------------------- Dates helpers ---------------------------

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


# --------------------------- Amadeus API ---------------------------

class RateLimitedError(Exception):
    pass

_amadeus_token: Optional[str] = None
_amadeus_token_expiry: Optional[datetime] = None
_flight_cache: Dict[Tuple[str, str, str, str], Tuple[dict, float]] = {}
_flight_cache_order: List[Tuple[str, str, str, str]] = []
_FLIGHT_CACHE_MAX = 30


def _cache_get(key: Tuple[str, str, str, str]) -> Optional[dict]:
    item = _flight_cache.get(key)
    if not item:
        return None
    payload, ts = item
    if time.time() - ts > 600:  # 10 mins TTL
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


def _parse_flight_offers_response(payload: dict, max_stops: int) -> List[Itinerary]:
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
        "- tap a chip (e.g. Cheaper, Non-stop)\n"
        "- type \"cheaper\" to try a lower budget\n"
        "- type \"fewer stops\" or \"non-stop only\"\n"
        "- type a different month, e.g. \"try July\" or \"move to September\"\n"
        "- or set a new budget directly, e.g. \"set budget to $600\"."
    )
    return "\n".join(lines)


# --------------------------- UI helpers ---------------------------

def load_bg_base64() -> Optional[str]:
    path = os.path.join('static', 'flight_bg.gif')
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('ascii')
    except Exception:
        return None


def inject_css(theme: str, bg_b64: Optional[str]):
    dark = theme != "light"
    palettes = {
        True: {
            "bg": "#050b1a",
            "panel": "#0f172a",
            "text": "#f8fafc",
            "muted": "#9ca3af",
            "border": "#1f2a40",
            "accent": "#38bdf8",
            "accent_alt": "#22c55e",
            "bot": "#131c33",
            "user": "linear-gradient(135deg, #22d3ee 0%, #818cf8 100%)",
        },
        False: {
            "bg": "#f5f7fb",
            "panel": "#ffffff",
            "text": "#0f172a",
            "muted": "#475569",
            "border": "#e2e8f0",
            "accent": "#2563eb",
            "accent_alt": "#10b981",
            "bot": "#f1f5ff",
            "user": "linear-gradient(135deg, #818cf8 0%, #c084fc 100%)",
        },
    }[dark]
    bg_layer = (
        f"background-image:url('data:image/gif;base64,{bg_b64}');"
        if bg_b64
        else ""
    )
    css = f"""
    <style>
        html, body {{
            background: {palettes["bg"]};
            color: {palettes["text"]};
            height: 100%;
            overflow: hidden;
        }}
        body::before {{
            content: '';
            position: fixed;
            inset: 0;
            opacity: {0.18 if dark else 0.12};
            {bg_layer}
            background-size: cover;
            background-position: center;
            pointer-events: none;
            z-index: -1;
        }}
        [data-testid="stAppViewContainer"] {{
            background: transparent;
        }}
        header, footer, [data-testid="stSidebar"] {{
            display: none;
        }}
        .main .block-container {{
            height: 100vh;
            padding: 1.25rem 1.5rem 1.5rem;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }}
        .flight-wrapper {{
            width: 100%;
            max-width: 1100px;
            margin: 0 auto;
            display: flex;
            flex-direction: column;
            gap: 0.9rem;
        }}
        .header-card {{
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
        }}
        .title-block {{
            color: {palettes["text"]};
        }}
        .title-block h1 {{
            margin: 0;
            font-size: 2rem;
        }}
        .title-block p {{
            margin: 0;
            color: {palettes["muted"]};
        }}
        .control-row {{
            display: flex;
            gap: 0.5rem;
            align-items: center;
        }}
        .control-row .stButton button {{
            border-radius: 999px;
            padding: 0.35rem 0.9rem;
            border: 1px solid {palettes["border"]};
            background: {palettes["panel"]};
            color: {palettes["text"]};
            font-weight: 600;
        }}
        .chip-panel {{
            background: {palettes["panel"]};
            border: 1px solid {palettes["border"]};
            border-radius: 20px;
            padding: 1rem 1.25rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, {0.35 if dark else 0.08});
        }}
        .chip-panel .chip-title {{
            font-weight: 600;
            margin-bottom: 0.5rem;
            color: {palettes["muted"]};
        }}
        .chip-panel .stButton>button {{
            width: 100%;
            border-radius: 14px;
            border: 1px solid {palettes["border"]};
            background: {palettes["bot"]};
            color: {palettes["text"]};
            font-weight: 600;
        }}
        .chip-panel .stButton>button:hover {{
            border-color: {palettes["accent"]};
            color: {palettes["accent"]};
        }}
        .chat-shell {{
            background: {palettes["panel"]};
            border-radius: 24px;
            border: 1px solid {palettes["border"]};
            padding: 1.5rem;
            box-shadow: 0 20px 45px rgba(0, 0, 0, {0.35 if dark else 0.08});
            display: flex;
            flex-direction: column;
            flex: 1;
            min-height: 420px;
            gap: 1rem;
        }}
        .chat-history {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            gap: 0.85rem;
        }}
        .chat-history::-webkit-scrollbar {{
            width: 6px;
        }}
        .chat-history::-webkit-scrollbar-thumb {{
            background: {palettes["border"]};
            border-radius: 4px;
        }}
        .chat-bubble {{
            padding: 0.9rem 1.1rem;
            border-radius: 18px;
            line-height: 1.4;
            font-size: 0.98rem;
            max-width: 88%;
        }}
        .bubble-title {{
            font-weight: 700;
            margin-bottom: 0.3rem;
        }}
        .chat-bubble.user {{
            background: {palettes["user"]};
            color: #ffffff;
            margin-left: auto;
            border-bottom-right-radius: 6px;
        }}
        .chat-bubble.bot {{
            background: {palettes["bot"]};
            color: {palettes["text"]};
            margin-right: auto;
            border-bottom-left-radius: 6px;
        }}
        .chat-bubble.error {{
            border: 1px solid {palettes["accent"]};
        }}
        .chat-bubble.greeting {{
            border: 1px dashed {palettes["accent_alt"]};
        }}
        .chat-input {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-top: 1px solid {palettes["border"]};
            padding-top: 0.75rem;
        }}
        .chat-input .stTextInput>div>div {{
            border-radius: 999px;
            border: 1px solid {palettes["border"]};
            background: {palettes["bot"]};
        }}
        .chat-input .stTextInput>div>div>input {{
            padding: 0.6rem 1.1rem;
            font-size: 1rem;
            color: {palettes["text"]};
            background: transparent;
        }}
        .chat-input .stButton button {{
            border-radius: 999px;
            height: 52px;
            width: 52px;
            border: none;
            font-size: 1.4rem;
            background: {palettes["accent"]};
            color: #ffffff;
        }}
        .chat-input .stButton button:hover {{
            filter: brightness(1.1);
        }}
        .mic-frame {{
            display: flex;
            justify-content: center;
            align-items: center;
            height: 52px;
        }}
        .mic-frame button {{
            height: 52px;
            width: 52px;
            border-radius: 999px;
            border: none;
            background: {palettes["accent_alt"]};
            color: #ffffff;
            font-size: 1.35rem;
            cursor: pointer;
        }}
        .mic-frame button.listening {{
            animation: pulse 1.4s infinite;
        }}
        @keyframes pulse {{
            0% {{ transform: scale(1); opacity: 1; }}
            50% {{ transform: scale(1.05); opacity: 0.7; }}
            100% {{ transform: scale(1); opacity: 1; }}
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def voice_component():
    st.components.v1.html(
        """
        <div class="mic-frame">
            <button id="airs-mic" type="button" title="Voice input">🎤</button>
        </div>
        <script>
            const micBtn = document.getElementById('airs-mic');
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {{
                micBtn.disabled = true;
                micBtn.title = 'Speech recognition not supported in this browser';
            }}
            let recognition = null;
            micBtn.addEventListener('click', () => {{
                if (!SpeechRecognition) return;
                if (recognition) {{
                    try {{ recognition.stop(); }} catch (err) {{}}
                    recognition = null;
                    micBtn.classList.remove('listening');
                    return;
                }}
                recognition = new SpeechRecognition();
                recognition.lang = 'en-US';
                recognition.interimResults = false;
                recognition.maxAlternatives = 1;
                micBtn.classList.add('listening');
                recognition.onresult = (event) => {{
                    const transcript = event.results[0][0].transcript;
                    const params = new URLSearchParams(window.location.search);
                    params.set('recognized', transcript);
                    window.location.search = params.toString();
                }};
                recognition.onend = () => {{
                    micBtn.classList.remove('listening');
                    recognition = null;
                }};
                recognition.onerror = () => {{
                    micBtn.classList.remove('listening');
                    recognition = null;
                }};
                recognition.start();
            }});
        </script>
        """,
        height=70,
    )


def submit_flight_text():
    text = st.session_state.flight_chat_text.strip()
    if text:
        st.session_state.pending_query = text


FLIGHT_GREETING = """
<div class="chat-bubble bot greeting">
    <div class="bubble-title">👋 Welcome to AI Chatbot!</div>
    <div>Hi! I am an AI assistant powered by Ollama open-source models.</div>
    <div>I can help you with questions, conversations, and more!</div>
</div>
"""


# --------------------------- App state ---------------------------

if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {role, text, error}
if "slots" not in st.session_state:
    st.session_state.slots = init_slots()
if "pending_slot" not in st.session_state:
    st.session_state.pending_slot = None
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None  # dict or None
if "theme" not in st.session_state:
    st.session_state.theme = 'dark'
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True
if "flight_chat_text" not in st.session_state:
    st.session_state.flight_chat_text = ""
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


def process_query(query: str):
    if not query:
        return
    st.session_state.show_greeting = False
    history: List[Dict] = st.session_state.history
    slots: Dict = st.session_state.slots
    pending_slot: Optional[str] = st.session_state.pending_slot
    last_intent_raw = st.session_state.last_intent
    last_intent: Optional[ParsedIntent] = None
    if last_intent_raw:
        try:
            last_intent = ParsedIntent(**last_intent_raw)
        except Exception:
            last_intent = None

    history.append({"role": "user", "text": query, "error": False})
    text_low = query.lower()

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

    slots = update_slots_from_text(slots, query)

    if pending_slot == "budget_total" and slots["budget_total"] is None:
        st.session_state.history.append({"role": "bot", "text": "I still didn't catch your budget. Tell me your total flight budget in USD.", "error": False})
    elif pending_slot == "trip_length" and (slots["min_days"] is None or slots["max_days"] is None):
        st.session_state.history.append({"role": "bot", "text": "I still need your trip length. For example: 4-7 days or for 5 days.", "error": False})
    elif pending_slot == "max_stops" and slots["max_stops"] is None:
        st.session_state.history.append({"role": "bot", "text": "How many stops at most are you okay with? 0 for non-stop, or 1, 2, etc.", "error": False})
    else:
        missing = next_missing_slot(slots)
        if missing is not None:
            st.session_state.pending_slot = missing
            st.session_state.history.append({"role": "bot", "text": SLOT_QUESTIONS[missing], "error": False})
        else:
            st.session_state.pending_slot = None
            intent = slots_to_intent(slots)
            if intent is None:
                err = "Something went wrong while assembling your request. Say 'reset' to start over."
                st.session_state.history.append({"role": "bot", "text": err, "error": True})
            else:
                date_from, date_to, min_nights, max_nights = build_search_window(intent)
                try:
                    with st.spinner("Searching flights…"):
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
                    st.session_state.history.append({"role": "bot", "text": summary, "error": False})
                    st.session_state.last_intent = intent.__dict__
                except Exception as e:
                    if isinstance(e, RateLimitedError):
                        err = (
                            "Amadeus rate limit reached. Please wait a bit and try again. "
                            "Tip: adjust your dates or try fewer quick tweaks in a row."
                        )
                    else:
                        err = f"Error calling Amadeus API: {e}"
                    st.session_state.history.append({"role": "bot", "text": err, "error": True})

    st.session_state.slots = slots


# --------------------------- UI Layout ---------------------------

st.set_page_config(page_title="AirScout — Flight Chatbot", page_icon="✈️", layout="wide")

bg_b64 = load_bg_base64()
theme = st.session_state.theme
inject_css(theme, bg_b64)

st.markdown('<div class="flight-wrapper">', unsafe_allow_html=True)

header_cols = st.columns([7, 1, 1])
with header_cols[0]:
    st.markdown(
        '<div class="title-block"><h1>AirScout</h1>'
        '<p>Plan smarter US trips with Ollama + Amadeus data</p></div>',
        unsafe_allow_html=True,
    )
with header_cols[1]:
    toggle_value = st.toggle("Dark mode", value=st.session_state.theme != "light")
    new_theme = "dark" if toggle_value else "light"
    if new_theme != st.session_state.theme:
        st.session_state.theme = new_theme
        st.rerun()
with header_cols[2]:
    if st.button("🧹", help="Reset conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.slots = init_slots()
        st.session_state.pending_slot = None
        st.session_state.last_intent = None
        st.session_state.show_greeting = True
        st.session_state.flight_chat_text = ""
        st.session_state.pending_query = None
        st.rerun()

pending_slot = st.session_state.pending_slot
chip_title = "Quick actions"
chips: List[str] = []
if pending_slot == "origin_code":
    chip_title = "Pick an origin"
    chips = [
        "I'm flying from New York City",
        "I'm flying from San Francisco",
        "I'm flying from Los Angeles",
        "I'm flying from Chicago",
    ]
elif pending_slot == "dest_code":
    chip_title = "Pick a destination"
    chips = ["to San Francisco", "to New York City", "to Miami", "to Seattle"]
elif pending_slot == "year":
    chip_title = "Pick a year"
    chips = ["in 2025", "in 2026"]
elif pending_slot == "budget_total":
    chip_title = "Select a budget"
    chips = ["budget $300", "budget $500", "budget $800"]
elif pending_slot == "trip_length":
    chip_title = "Trip length"
    chips = ["for 3-5 days", "for 5-7 days", "for 7-10 days"]
elif pending_slot == "max_stops":
    chip_title = "Stops preference"
    chips = ["non-stop only", "up to 1 stop", "up to 2 stops"]
else:
    chips = [
        "cheaper",
        "fewer stops",
        "non-stop only",
        "try July",
        "set budget to $600",
        "reset",
        "Example: I want to travel from New York City to San Francisco in June 2026. I can't spend more than $400 and want to stay 4-7 days. I'm okay with up to 2 stops.",
    ]

st.markdown('<div class="chip-panel">', unsafe_allow_html=True)
st.markdown(f'<div class="chip-title">{chip_title}</div>', unsafe_allow_html=True)
if chips:
    chip_cols = st.columns(min(4, len(chips)) or 1)
    for i, text in enumerate(chips):
        with chip_cols[i % len(chip_cols)]:
            if st.button(text, use_container_width=True, key=f"chip_{i}"):
                process_query(text)
                st.rerun()
else:
    st.caption("Start typing to begin planning.")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)
st.markdown('<div class="chat-history">', unsafe_allow_html=True)
if st.session_state.show_greeting and not st.session_state.history:
    st.markdown(FLIGHT_GREETING, unsafe_allow_html=True)

for msg in st.session_state.history:
    role = msg.get("role", "bot")
    classes = "chat-bubble bot" if role != "user" else "chat-bubble user"
    if msg.get("error"):
        classes += " error"
    safe_text = html.escape(msg.get("text", "")).replace("\n", "<br>")
    st.markdown(f'<div class="{classes}">{safe_text}</div>', unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="chat-input">', unsafe_allow_html=True)

input_cols = st.columns([9, 1, 1])
with input_cols[0]:
    st.text_input(
        "Tell me about your US trip…",
        key="flight_chat_text",
        placeholder="Tell me about your US trip…",
        label_visibility="collapsed",
        on_change=submit_flight_text,
    )
with input_cols[1]:
    voice_component()
with input_cols[2]:
    if st.button("⬆️", key="flight_send", help="Send message"):
        submit_flight_text()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Handle recognized text from voice component via query param
recognized = None
try:
    recognized = st.experimental_get_query_params().get("recognized", [None])[0]
except Exception:
    try:
        recognized = st.query_params.get("recognized")
    except Exception:
        recognized = None
if recognized:
    process_query(recognized)
    try:
        st.experimental_set_query_params()
    except Exception:
        pass
    st.rerun()

pending_text = st.session_state.pending_query
if pending_text:
    st.session_state.pending_query = None
    cleaned = pending_text.strip()
    if cleaned:
        if cleaned.lower() in {"reset", "start over"}:
            st.session_state.history = []
            st.session_state.slots = init_slots()
            st.session_state.pending_slot = None
            st.session_state.last_intent = None
            st.session_state.show_greeting = True
        else:
            process_query(cleaned)
        st.session_state.flight_chat_text = ""
        st.rerun()
