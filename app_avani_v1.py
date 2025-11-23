import base64
import calendar
import html
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
import scipy.io.wavfile as wavfile
import sounddevice as sd
import streamlit as st
from dotenv import find_dotenv, load_dotenv
from gtts import gTTS

try:
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass

try:
    from faster_whisper import WhisperModel
    _WHISPER_AVAILABLE = True
    _WHISPER_IMPORT_ERROR = ""
except Exception as exc:
    WhisperModel = None  # type: ignore
    _WHISPER_AVAILABLE = False
    _WHISPER_IMPORT_ERROR = str(exc)

# ---------------------------------------------------------------------------
# Amadeus config + domain data
# ---------------------------------------------------------------------------

AMADEUS_CLIENT_ID = os.environ.get("AMADEUS_CLIENT_ID")
AMADEUS_CLIENT_SECRET = os.environ.get("AMADEUS_CLIENT_SECRET")
AMADEUS_AUTH_URL = "https://test.api.amadeus.com/v1/security/oauth2/token"
AMADEUS_FLIGHT_OFFERS_URL = "https://test.api.amadeus.com/v2/shopping/flight-offers"
_amadeus_token: Optional[str] = None
_amadeus_token_expiry: Optional[datetime] = None

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
    "budget_total": "What is your total flight budget in USD?",
    "trip_length": "How many days do you want to stay? For example: 4-7 days.",
    "max_stops": "What is the maximum number of stops you are okay with? (0 = non-stop, 1, 2, etc.)",
}

SLOT_DESCRIPTIONS = {
    "origin_code": "departure city or airport",
    "dest_code": "destination city or airport",
    "budget_total": "total flight budget in USD",
    "trip_length": "trip length in days",
    "min_days": "minimum number of days for the stay",
    "max_days": "maximum number of days for the stay",
    "max_stops": "preferred maximum number of stops",
    "year": "travel year",
    "month": "travel month",
}


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
    candidates = {
        date_from,
        date_from + timedelta(days=max(0, window_days // 2)),
        date_from + timedelta(days=max(0, window_days - max_nights)),
    }
    durations = sorted({min_nights, max_nights, (min_nights + max_nights) // 2})
    for dep in sorted(d for d in candidates if d <= date_to):
        for stay in durations:
            ret = dep + timedelta(days=stay)
            if ret <= date_to:
                pairs.append((dep, ret))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs


def _get_amadeus_token() -> str:
    global _amadeus_token, _amadeus_token_expiry
    if (
        _amadeus_token
        and _amadeus_token_expiry
        and datetime.now(timezone.utc) < _amadeus_token_expiry
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
    _amadeus_token_expiry = datetime.now(timezone.utc) + timedelta(seconds=expires_in - 60)
    return token


def _parse_iso_duration_hours(duration: str) -> float:
    if not duration or not duration.startswith("PT"):
        return 0.0
    hours = 0
    minutes = 0
    mh = re.search(r"(\d+)H", duration)
    mm = re.search(r"(\d+)M", duration)
    if mh:
        hours = int(mh.group(1))
    if mm:
        minutes = int(mm.group(1))
    return hours + minutes / 60.0


def _parse_flight_offers(payload: dict, max_stops: int) -> List[Itinerary]:
    itineraries: List[Itinerary] = []
    data = payload.get("data", [])
    for item in data:
        price_block = item.get("price", {})
        total_str = price_block.get("total")
        if not total_str:
            continue
        try:
            price = float(total_str)
        except ValueError:
            continue
        currency = price_block.get("currency", "USD")
        itineraries_raw = item.get("itineraries", [])
        if len(itineraries_raw) < 2:
            continue
        out_it, in_it = itineraries_raw[0], itineraries_raw[1]
        out_segments = out_it.get("segments", [])
        in_segments = in_it.get("segments", [])
        if not out_segments or not in_segments:
            continue
        stops_out = max(0, len(out_segments) - 1)
        stops_in = max(0, len(in_segments) - 1)
        if stops_out > max_stops or stops_in > max_stops:
            continue
        out_depart = out_segments[0].get("departure", {}).get("at")
        in_arrive = in_segments[-1].get("arrival", {}).get("at")
        if not out_depart or not in_arrive:
            continue
        try:
            out_date = date.fromisoformat(out_depart.split("T")[0])
            in_date = date.fromisoformat(in_arrive.split("T")[0])
        except Exception:
            continue
        dur_out = _parse_iso_duration_hours(out_it.get("duration", ""))
        dur_in = _parse_iso_duration_hours(in_it.get("duration", ""))
        total_hours = dur_out + dur_in
        carrier = out_segments[0].get("carrierCode", "")
        flight_number = out_segments[0].get("number", "")
        summary = (
            f"{price:.0f} {currency}, {out_date.isoformat()} → {in_date.isoformat()}, "
            f"{stops_out} stops out / {stops_in} back, ≈{total_hours:.1f}h, Carrier {carrier} {flight_number}"
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
    results: List[Itinerary] = []
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
        itineraries = _parse_flight_offers(resp.json(), max_stops=max_stops)
        itineraries = [it for it in itineraries if it.price <= budget_total]
        results.extend(itineraries)
        if len(results) >= 10:
            break
    return results


def summarize_itineraries(intent: ParsedIntent, itineraries: List[Itinerary]) -> str:
    if not itineraries:
        return (
            f"I couldn't find itineraries for {intent.origin_code} → {intent.dest_code} "
            f"in {intent.year} within about ${intent.budget_total:.0f} "
            f"for a {intent.min_days}-{intent.max_days} day trip and up to {intent.max_stops} stops.\n"
            "Consider increasing budget, allowing more stops, or shifting dates."
        )
    itineraries_sorted = sorted(itineraries, key=lambda x: x.price)
    unique: List[Itinerary] = []
    seen = set()
    for it in itineraries_sorted:
        key = (
            round(it.price, 2),
            it.outbound_date,
            it.inbound_date,
            it.stops_outbound,
            it.stops_inbound,
            it.summary,
        )
        if key not in seen:
            seen.add(key)
            unique.append(it)
        if len(unique) >= 3:
            break
    cheapest = unique[0]
    lines = [
        f"Here are options for {intent.origin_code} → {intent.dest_code} "
        f"in {intent.year}, budget around ${intent.budget_total:.0f}, "
        f"{intent.min_days}-{intent.max_days} days, up to {intent.max_stops} stops:\n"
    ]
    for idx, it in enumerate(unique, start=1):
        label = " (cheapest)" if it is cheapest else ""
        lines.append(
            f"{idx}) Price: ~${it.price:.0f} {it.currency}{label}\n"
            f"   Dates: {it.outbound_date.isoformat()} → {it.inbound_date.isoformat()}\n"
            f"   Stops: {it.stops_outbound} out / {it.stops_inbound} back\n"
            f"   Time: ≈{it.total_duration_hours:.1f}h total\n"
            f"   {it.summary}\n"
        )
    lines.append(
        "Want to tweak it? Try 'cheaper', 'non-stop only', 'allow 2 stops', or a new month like 'try July'."
    )
    return "\n".join(lines)


def stylize_itinerary_summary(intent: ParsedIntent, summary_text: str) -> str:
    """Run the structured itinerary summary through Ollama for a conversational tone."""
    description = describe_intent(intent)
    prompt = (
        "You are AirScout, a concise but friendly travel assistant. "
        "Given the structured flight summary below and the trip description, rewrite it as a warm, natural response. "
        "Open with a short greeting that highlights the top option, then lay out each choice as a numbered sentence (e.g., '1) ...', '2) ...') using plain text—no bold or asterisks. "
        "Keep the tone encouraging and conversational, and close with a simple suggestion for next steps.\n"
        f"Trip description: {description}\n"
        f"Structured summary:\n{summary_text}\n"
        "Assistant response:"
    )
    return respond_with_ollama(prompt, summary_text)


def fallback_itineraries(intent: ParsedIntent) -> List[Itinerary]:
    samples = [
        {"price": 219, "hours": 11.9, "stops": (0, 0), "carrier": "AA", "flight": "1456"},
        {"price": 248, "hours": 12.4, "stops": (1, 1), "carrier": "DL", "flight": "345"},
        {"price": 265, "hours": 13.1, "stops": (1, 1), "carrier": "UA", "flight": "987"},
    ]
    travel_year = intent.year or datetime.now().year
    fallback_year = min(travel_year, datetime.now().year + 1)
    month = intent.month or 6
    start_day = 1
    itineraries: List[Itinerary] = []
    for sample in samples:
        outbound = date(fallback_year, month, start_day)
        inbound = outbound + timedelta(days=max(intent.min_days or 4, 4))
        summary = (
            f"{sample['price']:.0f} USD, {outbound.isoformat()} → {inbound.isoformat()}, "
            f"{sample['stops'][0]} stops out / {sample['stops'][1]} back, "
            f"≈{sample['hours']:.1f}h, Carrier {sample['carrier']} {sample['flight']}"
        )
        itineraries.append(
            Itinerary(
                price=sample["price"],
                currency="USD",
                outbound_date=outbound,
                inbound_date=inbound,
                total_duration_hours=sample["hours"],
                stops_outbound=sample["stops"][0],
                stops_inbound=sample["stops"][1],
                summary=summary,
            )
        )
        start_day += 3
    return itineraries


st.set_page_config(
    page_title="AirScout",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True
if "audio_playing" not in st.session_state:
    st.session_state.audio_playing = {}
if "chat_text" not in st.session_state:
    st.session_state.chat_text = ""
if st.session_state.get("reset_chat_text"):
    st.session_state.chat_text = ""
    st.session_state.pop("reset_chat_text", None)
prefill_text = st.session_state.pop("prefill_chat_text", None)
if prefill_text:
    st.session_state.chat_text = prefill_text
if "slots" not in st.session_state:
    st.session_state.slots = init_slots()
if "pending_slot" not in st.session_state:
    st.session_state.pending_slot = None
if "last_intent" not in st.session_state:
    st.session_state.last_intent = None
if "pending_intent" not in st.session_state:
    st.session_state.pending_intent = None
if "awaiting_confirmation" not in st.session_state:
    st.session_state.awaiting_confirmation = False

def get_dynamic_suggestions() -> list[tuple[str, str]]:
    if st.session_state.awaiting_confirmation:
        return [
            ("✅ Yes", "yes"),
            ("❌ No", "no"),
            ("Reset", "reset"),
        ]
    if st.session_state.get("last_intent"):
        return [
            ("Cheaper", "cheaper"),
            ("Non-stop only", "non-stop only"),
            ("Allow 2 stops", "allow 2 stops"),
            ("Try July", "try July"),
        ]
    slots = st.session_state.slots
    if slots_complete(slots):
        return [("Confirm", "yes please confirm"), ("Adjust budget", "set my budget to $500"), ("Reset", "reset")]
    pending = st.session_state.pending_slot or find_next_missing_slot(slots)

    def fmt(label: str, text: str) -> tuple[str, str]:
        return (label, text)

    if pending == "origin_code":
        return [
            fmt("From NYC", "I'm flying from New York City"),
            fmt("From SFO", "I'm flying from San Francisco"),
            fmt("From LAX", "I'm flying from Los Angeles"),
            fmt("From ORD", "I'm flying from Chicago"),
        ]
    if pending == "dest_code":
        return [
            fmt("To Miami", "to Miami"),
            fmt("To Seattle", "to Seattle"),
            fmt("To Boston", "to Boston"),
            fmt("To LA", "to Los Angeles"),
        ]
    if pending == "year":
        return [fmt("Travel 2025", "I'm traveling in 2025"), fmt("Travel 2026", "I'm traveling in 2026")]
    if pending == "budget_total":
        return [
            fmt("Budget $300", "My total flight budget is $300"),
            fmt("Budget $450", "I can spend around $450"),
            fmt("Budget $600", "Set my budget to $600"),
        ]
    if pending == "trip_length":
        return [
            fmt("3-5 days", "I want to stay for 3-5 days"),
            fmt("5-7 days", "Plan for 5-7 days"),
            fmt("7-10 days", "I can stay for 7-10 days"),
        ]
    if pending == "max_stops":
        return [
            fmt("Non-stop only", "I'd like non-stop flights"),
            fmt("Up to 1 stop", "I'm ok with up to 1 stop"),
            fmt("Up to 2 stops", "Allow up to 2 stops"),
        ]

    return [
        fmt("Cheaper", "Can we find cheaper options?"),
        fmt("New month", "Try searching in July instead."),
        fmt("More stops", "Allow more stops if it helps."),
        fmt("Reset", "reset"),
    ]

GREETING_HTML = """
<div class="chat-bubble bot-bubble greeting">
    <div class="bubble-title">👋 Welcome to AirScout!</div>
    <div>I use Amadeus flight data to plan US trips within your budget.</div>
    <div>Tell me where you're flying from, to, dates, budget, and stops.</div>
</div>
"""

ASSISTANT_PROMPT = (
    "You are AirScout, a helpful US flight-planning assistant. Ask follow-up questions "
    "to capture origin airport, destination, travel timing, trip length, budget, and max "
    "stops. When provided with itinerary summaries, translate them into concise, friendly "
    "messages that highlight key options and suggest next steps."
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:1b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")


def call_ollama(prompt: str, user_message: str = "", model: str | None = None) -> str:
    """Call Ollama locally with error handling."""
    model = model or OLLAMA_MODEL
    url = f"{OLLAMA_URL}/api/generate"
    full_prompt = f"{prompt}\n\nUser: {user_message}\nAssistant:" if user_message else prompt
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
    }
    try:
        response = requests.post(url, json=payload, timeout=(10, 60))
        response.raise_for_status()
        data = response.json()
        return data.get("response", "No response generated.")
    except requests.exceptions.Timeout:
        return "The model took longer than expected. Please try again in a moment."
    except requests.exceptions.ConnectionError as exc:
        return f"Could not connect to Ollama at {OLLAMA_URL}: {exc}"
    except Exception as exc:
        return f"Ollama error: {exc}"


def _ollama_failed_response(text: Optional[str]) -> bool:
    if not text:
        return True
    lowered = text.lower()
    failure_markers = [
        "ollama error",
        "could not connect",
        "took longer than expected",
        "please try again",
        "error:",
    ]
    return any(marker in lowered for marker in failure_markers)


def _clean_ollama_text(text: str) -> str:
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    return text


def respond_with_ollama(prompt: str, fallback: str) -> str:
    """Helper to call Ollama and gracefully fall back to static text."""
    raw = call_ollama(prompt)
    if _ollama_failed_response(raw):
        return fallback
    return _clean_ollama_text(raw)


def text_to_speech(text: str, idx: int) -> str | None:
    """Convert text to speech and return a temporary mp3 path."""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as exc:
        st.error(f"Text-to-speech error: {exc}")
        return None


@st.cache_resource
def load_whisper_model() -> WhisperModel | None:
    """Load Faster-Whisper base model once."""
    if not _WHISPER_AVAILABLE:
        return None
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as exc:
        st.error(
            "Could not load Faster-Whisper. Make sure torch and faster-whisper are installed. "
            f"Details: {exc}"
        )
        return None


def speech_to_text() -> str | None:
    """Capture microphone audio, transcribe via Faster-Whisper, and return text."""
    whisper_model = load_whisper_model()
    if whisper_model is None:
        if _WHISPER_AVAILABLE:
            st.error("Faster-Whisper model is not available. Please check your installation.")
        else:
            st.error(
                "The speech-to-text component could not load faster-whisper. "
                "Install it with: pip install faster-whisper torch torchvision torchaudio\n"
                f"Details: {_WHISPER_IMPORT_ERROR}"
            )
        return None

    duration = 10
    sample_rate = 16000
    try:
        st.info(f"🎙️ Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()
        st.success("✅ Audio captured! Transcribing...")

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavfile.write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))

        segments, _ = whisper_model.transcribe(temp_file.name, beam_size=5, language="en")
        transcription = " ".join(segment.text for segment in segments).strip()

        try:
            os.unlink(temp_file.name)
        except OSError:
            pass

        if transcription:
            return transcription
        st.warning("No speech detected. Please try again.")
    except Exception as exc:
        st.error(f"Speech recognition error: {exc}")
    return None


def clear_audio_buffers():
    for path in st.session_state.audio_playing.values():
        try:
            os.unlink(path)
        except OSError:
            pass
    st.session_state.audio_playing = {}


def submit_text_from_box():
    text = st.session_state.chat_text.strip()
    if text:
        st.session_state.text_to_send = text


def reset_conversation():
    clear_audio_buffers()
    st.session_state.slots = init_slots()
    st.session_state.pending_slot = None
    st.session_state.last_intent = None
    st.session_state.pending_intent = None
    st.session_state.awaiting_confirmation = False
    st.session_state.messages = []
    st.session_state.show_greeting = True
    st.session_state.reset_chat_text = True


def _load_last_intent() -> ParsedIntent | None:
    data = st.session_state.get("last_intent")
    if not data:
        return None
    try:
        return ParsedIntent(**data)
    except Exception:
        return None


def _store_last_intent(intent: ParsedIntent | None):
    if intent is None:
        st.session_state.last_intent = None
    else:
        st.session_state.last_intent = intent.__dict__


def ask_user(slot_key: str, slots: dict):
    fallback = SLOT_QUESTIONS.get(slot_key, "Could you share a few more trip details?")
    slot_desc = SLOT_DESCRIPTIONS.get(slot_key, slot_key)
    known_values = {k: v for k, v in slots.items() if v is not None}
    prompt = (
        "You are AirScout, a conversational US flight-planning assistant. "
        "Review the known trip details and craft ONE short question in second person "
        "to gather the missing information.\n"
        f"Known details: {json.dumps(known_values)}\n"
        f"Missing information: {slot_desc} (slot key: {slot_key}). "
        "Ask naturally, avoid bullet points, stay under 25 words."
    )
    question = respond_with_ollama(prompt, fallback)
    st.session_state.messages.append({"role": "assistant", "content": question})


def _extract_json_block(text: str) -> Optional[dict]:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def _safe_int(value) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(float(value))
    except Exception:
        return None


def _city_to_code(city: Optional[str]) -> Optional[str]:
    if not city:
        return None
    city = city.lower().strip()
    return CITY_TO_AIRPORT.get(city, city.upper() if len(city) == 3 else None)


def parse_slots_with_ollama(user_text: str, slots: dict) -> dict:
    known = {
        "origin_code": slots.get("origin_code"),
        "dest_code": slots.get("dest_code"),
        "year": slots.get("year"),
        "month": slots.get("month"),
        "min_days": slots.get("min_days"),
        "max_days": slots.get("max_days"),
        "budget_total": slots.get("budget_total"),
        "max_stops": slots.get("max_stops"),
    }
    extraction_prompt = (
        "Extract flight-planning slots from the traveler. "
        "Return ONLY JSON with keys origin_city, destination_city, year, month_number, "
        "month_name, budget_total, min_days, max_days, max_stops. Use null for unknown fields.\n"
        f"Current known values: {json.dumps(known)}\n"
        f"Traveler input: {user_text}\n"
        "JSON:"
    )
    raw = call_ollama(extraction_prompt, "")
    data = _extract_json_block(raw or "")
    return data or {}


def apply_llm_slots(slots: dict, llm_data: dict) -> dict:
    origin_city = llm_data.get("origin_city")
    dest_city = llm_data.get("destination_city")
    year = _safe_int(llm_data.get("year"))
    month_number = _safe_int(llm_data.get("month_number"))
    month_name = llm_data.get("month_name")
    budget = llm_data.get("budget_total")
    min_days = _safe_int(llm_data.get("min_days"))
    max_days = _safe_int(llm_data.get("max_days"))
    max_stops = _safe_int(llm_data.get("max_stops"))

    if origin_city:
        code = _city_to_code(origin_city)
        if code:
            slots["origin_code"] = code
    if dest_city:
        code = _city_to_code(dest_city)
        if code:
            slots["dest_code"] = code
    if year:
        slots["year"] = year
    if month_number and 1 <= month_number <= 12:
        slots["month"] = month_number
    elif month_name:
        slots["month"] = MONTHS.get(month_name.lower(), slots.get("month"))
    if budget is not None:
        try:
            slots["budget_total"] = float(budget)
        except Exception:
            pass
    if min_days is not None and max_days is not None:
        slots["min_days"] = min_days
        slots["max_days"] = max_days
    if max_stops is not None:
        slots["max_stops"] = max(0, max_stops)
    return slots


def describe_intent(intent: ParsedIntent) -> str:
    month_text = (
        calendar.month_name[intent.month]
        if intent.month and 1 <= intent.month <= 12
        else "flexible months"
    )
    return (
        f"from {intent.origin_code} to {intent.dest_code}"
        f" around {intent.year} ({month_text}), staying {intent.min_days}-{intent.max_days} days,"
        f" budget about ${intent.budget_total:.0f}, up to {intent.max_stops} stops"
    )


def confirm_intent(intent: ParsedIntent):
    _store_pending_intent(intent)
    st.session_state.awaiting_confirmation = True
    summary = describe_intent(intent)
    fallback = (
        "I believe I have everything I need. "
        f"I'm planning to search flights {summary}. "
        "Reply YES to proceed or NO to adjust anything."
    )
    prompt = (
        "You are AirScout, a friendly travel assistant. "
        "You have collected these flight search details:\n"
        f"{summary}\n"
        "Write the final confirmation message directly—no preface, no quotes, no meta commentary. "
        "Restate the key facts in one or two sentences and end with a request for the traveler to reply YES to proceed or NO to adjust."
    )
    confirmation_text = respond_with_ollama(prompt, fallback)
    st.session_state.messages.append({"role": "assistant", "content": confirmation_text})


def run_intent_search(intent: ParsedIntent):
    with st.spinner("✈️ Searching Amadeus offers..."):
        try:
            date_from, date_to, min_nights, max_nights = build_search_window(intent)
            itineraries = call_flight_api(
                origin=intent.origin_code,
                dest=intent.dest_code,
                date_from=date_from,
                date_to=date_to,
                min_nights=min_nights,
                max_nights=max_nights,
                max_stops=intent.max_stops,
                budget_total=intent.budget_total,
            )
            summary = summarize_itineraries(intent, itineraries)
            summary = stylize_itinerary_summary(intent, summary)
            st.session_state.messages.append({"role": "assistant", "content": summary})
            _store_last_intent(intent)
        except Exception as exc:
            fallback = fallback_itineraries(intent)
            if fallback:
                summary = summarize_itineraries(intent, fallback)
                summary = stylize_itinerary_summary(intent, summary)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": summary + "\n\n_(Amadeus sandbox was unavailable; showing example options.)_",
                    }
                )
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": f"I couldn't complete the search ({exc}). Try adjusting the details or wait a moment.",
                    }
                )


def _normalize_tokens(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def handle_confirmation_response(user_text: str) -> str:
    text = user_text.strip().lower()
    tokens = _normalize_tokens(user_text)

    yes_phrases = [
        "yes",
        "yeah",
        "yup",
        "sure",
        "please do",
        "go ahead",
        "confirm",
        "proceed",
        "sounds good",
        "do it",
        "okay go",
    ]
    no_phrases = [
        "no",
        "nope",
        "nah",
        "stop",
        "wait",
        "hold on",
        "not yet",
        "don't",
    ]

    def _has_phrase(phrases: List[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    if text in {"y", "yes"} or _has_phrase(yes_phrases) or any(token in {"yes", "y", "confirm"} for token in tokens):
        intent = _load_pending_intent()
        st.session_state.awaiting_confirmation = False
        _store_pending_intent(None)
        if intent is None:
            st.session_state.messages.append(
                {"role": "assistant", "content": "I lost the latest trip details. Please repeat them."}
            )
            return "handled"
        run_intent_search(intent)
        return "handled"

    if text in {"n", "no"} or _has_phrase(no_phrases) or any(token in {"no", "stop"} for token in tokens):
        st.session_state.awaiting_confirmation = False
        _store_pending_intent(None)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "No problem! Tell me which detail you'd like to change—origin, destination, dates, budget, or stops.",
            }
        )
        return "handled"

    return "continue"


def apply_direct_city_hints(slots: dict, text: str) -> dict:
    low = text.lower()
    for city, code in CITY_TO_AIRPORT.items():
        token = re.escape(city.lower())
        if re.search(rf"\bfrom\s+{token}\b", low):
            slots["origin_code"] = code
        if re.search(rf"\bto\s+{token}\b", low):
            slots["dest_code"] = code
    return slots


def find_next_missing_slot(slots: dict) -> str | None:
    if slots.get("origin_code") is None:
        return "origin_code"
    if slots.get("dest_code") is None:
        return "dest_code"
    if slots.get("budget_total") is None:
        return "budget_total"
    if slots.get("min_days") is None or slots.get("max_days") is None:
        return "trip_length"
    if slots.get("max_stops") is None:
        return "max_stops"
    return None


def slots_complete(slots: dict) -> bool:
    return (
        slots.get("origin_code")
        and slots.get("dest_code")
        and slots.get("budget_total") is not None
        and slots.get("min_days") is not None
        and slots.get("max_days") is not None
        and slots.get("max_stops") is not None
    )


def _load_pending_intent() -> ParsedIntent | None:
    data = st.session_state.get("pending_intent")
    if not data:
        return None
    try:
        return ParsedIntent(**data)
    except Exception:
        return None


def _store_pending_intent(intent: ParsedIntent | None):
    if intent is None:
        st.session_state.pending_intent = None
    else:
        st.session_state.pending_intent = intent.__dict__


def process_user_message(user_text: str):
    slots = st.session_state.slots
    pending_slot = st.session_state.pending_slot
    last_intent = _load_last_intent()

    text_low = user_text.lower()

    if st.session_state.awaiting_confirmation:
        confirmation_result = handle_confirmation_response(user_text)
        if confirmation_result == "handled":
            return
        st.session_state.awaiting_confirmation = False
        _store_pending_intent(None)

    if text_low in {"reset", "start over"}:
        reset_conversation()
        return

    if last_intent:
        if "cheaper" in text_low:
            slots["budget_total"] = max(50.0, float(last_intent.budget_total) * 0.9)
        if any(term in text_low for term in ["non-stop", "nonstop", "non stop", "direct"]):
            slots["max_stops"] = 0
        elif "fewer stops" in text_low:
            slots["max_stops"] = max(0, last_intent.max_stops - 1)
        elif "more stops" in text_low:
            slots["max_stops"] = last_intent.max_stops + 1

        new_month = extract_month(text_low)
        if new_month:
            slots["month"] = new_month

        new_budget = extract_budget(user_text)
        if new_budget is not None:
            slots["budget_total"] = float(new_budget)

    slots = apply_direct_city_hints(slots, user_text)
    slots = update_slots_from_text(slots, user_text)

    llm_data = parse_slots_with_ollama(user_text, slots)
    if llm_data:
        slots = apply_llm_slots(slots, llm_data)

    # if destination accidentally mirrors origin (e.g., LLM guessed), clear it so we re-ask
    if slots.get("origin_code") and slots.get("dest_code") == slots.get("origin_code"):
        slots["dest_code"] = None

    if slots_complete(slots):
        st.session_state.pending_slot = None
        if slots.get("year") is None:
            slots["year"] = datetime.now().year
        intent = slots_to_intent(slots)
        if intent is None:
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": "Something went wrong while assembling the request. Say 'reset' to start over.",
                }
            )
        else:
            confirm_intent(intent)
            st.session_state.slots = slots
            return

    missing = find_next_missing_slot(slots)
    if missing is not None:
        st.session_state.pending_slot = missing
        ask_user(missing, slots)
        st.session_state.slots = slots
        return

    st.session_state.slots = slots


def autoplay_audio(file_path: str):
    """Embed an autoplaying audio element without showing controls."""
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        audio_html = f"""
        <audio autoplay hidden>
            <source src="data:audio/mp3;base64,{encoded}" type="audio/mp3" />
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception:
        st.audio(file_path, format="audio/mp3")


def build_css(dark_mode: bool) -> str:
    palettes = {
        True: {
            "bg": "#03030a",
            "panel": "#090d1a",
            "muted": "#b6c2ff",
            "text": "#e5fbff",
            "border": "#1c2340",
            "user": "linear-gradient(135deg, #12d3ff 0%, #7156ff 100%)",
            "bot": "#0f1527",
            "greeting": "#121a34",
            "input_bg": "#0d1426",
            "accent": "#20ffb2",
            "accent_text": "#071b17",
            "chip_hover": "#20ffb2",
        },
        False: {
            "bg": "#f8fbff",
            "panel": "#ffffff",
            "muted": "#445063",
            "text": "#111827",
            "border": "#d5d9e7",
            "user": "linear-gradient(135deg, #5d5fef 0%, #bb79ff 100%)",
            "bot": "#f2f5ff",
            "greeting": "#e8ecff",
            "input_bg": "#f6f7fb",
            "accent": "#0c8b6f",
            "accent_text": "#ffffff",
            "chip_hover": "#0c8b6f",
        },
    }[dark_mode]

    button_color_fix = ""
    if dark_mode:
        button_color_fix = """
        /* Fix button text color in dark mode */
        [data-testid="baseButton"] > div,
        .stButton > button,
        button[kind="secondary"],
        body button {
            color: #0A1A2F !important;
        }
        """

    return f"""
    <style>
        {button_color_fix}
        html, body {{
            background: {palettes["bg"]};
            color: {palettes["text"]};
            height: 100%;
            overflow: hidden;
        }}
        [data-testid="stAppViewContainer"] {{
            background: {palettes["bg"]};
            color: {palettes["text"]};
        }}
        header, footer, [data-testid="stSidebar"] {{
            display: none;
        }}
        .main .block-container {{
            padding: 0;
            min-height: 100vh;
        }}
        .chat-wrapper {{
            width: min(900px, calc(100% - 2rem));
            position: fixed;
            bottom: 1.25rem;
            left: 50%;
            transform: translateX(-50%);
            z-index: 4;
        }}
        .sticky-header {{
            position: fixed;
            top: 1rem;
            left: 50%;
            transform: translateX(-50%);
            width: min(900px, calc(100% - 2rem));
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 1rem;
            padding: 0.75rem 1.25rem;
            background: {palettes["panel"]};
            border: 1px solid {palettes["border"]};
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.25);
            z-index: 6;
        }}
        .hero-title {{
            font-size: 2.2rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
            color: {palettes["text"]};
            text-align: center;
        }}
        .hero-sub {{
            color: {palettes["muted"]};
            font-size: 0.95rem;
            margin: 0;
            text-align: center;
        }}
        .chat-history {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            gap: 1.8rem;
            padding-right: 0.25rem;
        }}
        .chat-history::-webkit-scrollbar {{
            width: 6px;
        }}
        .chat-history::-webkit-scrollbar-thumb {{
            background: {palettes["border"]};
            border-radius: 4px;
        }}
        .chat-bubble {{
            padding: 0.95rem 1.1rem;
            border-radius: 20px;
            line-height: 1.45;
            font-size: 0.98rem;
            max-width: 85%;
            margin: 0.9rem 0;
        }}
        .chat-bubble.user-bubble {{
            background: {palettes["user"]};
            color: #ffffff;
            margin-left: auto;
            border-bottom-right-radius: 6px;
        }}
        .chat-bubble.bot-bubble {{
            background: {palettes["bot"]};
            color: {palettes["text"]};
            margin-right: auto;
            border-bottom-left-radius: 6px;
        }}
        .chat-bubble.greeting {{
            background: {palettes["greeting"]};
            border: 1px dashed {palettes["border"]};
        }}
        .bubble-title {{
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .chat-input {{
            display: flex;
            align-items: center;
            gap: 0.85rem;
            margin-top: 1rem;
            border-top: 1px solid {palettes["border"]};
            padding-top: 0.75rem;
        }}
        .prompt-chips {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }}
        .prompt-chips button {{
            border-radius: 999px !important;
            border: 1px solid {palettes["border"]} !important;
            background: {'#E6ECFF' if dark_mode else '#00008B'} !important;
            color: #0A1A2F !important;
            font-weight: 600 !important;
            padding: 0.35rem 0.9rem !important;
            box-shadow: 0 6px 18px rgba(0,0,0,{0.35 if dark_mode else 0.15}) !important;
        }}
        .prompt-chips button span {{
            color: #0A1A2F !important;
            font-weight: 600 !important;
        }}
        .prompt-chips button:hover {{
            border-color: {palettes["chip_hover"]} !important;
            color: {palettes["chip_hover"]} !important;
        }}
        .toggle-text {{
            text-align: center;
            font-weight: 600;
            color: {palettes["text"]};
        }}
        .toggle-inline {{
            display: flex;
            align-items: center;
            height: 100%;
            padding-left: 0.3rem;
        }}
        .dark-toggle-wrapper {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
        }}
        .dark-toggle-wrapper [data-testid="stToggle"] {{
            margin: 0 !important;
        }}
        .dark-toggle-wrapper [data-testid="stToggle"] span {{
            display: none;
        }}
        .dark-toggle-wrapper [data-testid="stToggle"] label {{
            position: relative;
            width: 80px;
            height: 34px;
            border-radius: 999px;
            border: 1px solid {palettes["border"]};
            background: {("#0f1527" if dark_mode else "#f0f3ff")};
            display: inline-flex;
            justify-content: center;
            align-items: center;
        }}
        .dark-toggle-wrapper [data-testid="stToggle"] label:before {{
            content: "☀️";
            position: absolute;
            left: 8px;
            font-size: 16px;
            opacity: {0.3 if dark_mode else 1};
            transition: opacity 0.2s ease;
        }}
        .dark-toggle-wrapper [data-testid="stToggle"] label:after {{
            content: "🌙";
            position: absolute;
            right: 8px;
            font-size: 16px;
            opacity: {1 if dark_mode else 0.3};
            transition: opacity 0.2s ease;
        }}
        .chat-input .stTextInput > div > div {{
            border-radius: 999px;
            border: 1px solid {palettes["border"]};
            background: {palettes["input_bg"]};
        }}
        .chat-input .stTextInput > div > div > input {{
            padding: 0.65rem 1.2rem;
            font-size: 1rem;
            background: transparent;
            color: {palettes["text"]};
        }}
        .chat-input .stButton button {{
            width: 54px;
            height: 54px;
            border-radius: 999px;
            border: none;
            font-size: 1.45rem;
            background: {palettes["accent"]};
            color: {palettes["accent_text"]};
        }}
        .chat-input .stButton button:hover {{
            filter: brightness(1.1);
        }}
        .tts-controls button {{
            border-radius: 50px;
            border: 1px solid {palettes["border"]};
            background: {palettes["bot"]};
            color: {palettes["text"]};
            font-size: 0.9rem;
            width: 100%;
        }}
    </style>
    """


st.markdown(build_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------

st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
st.markdown('<div class="sticky-header">', unsafe_allow_html=True)
top_cols = st.columns([7, 1])
with top_cols[0]:
    st.markdown('<div class="hero-title">AirScout</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub">A conversational flight planning application.</p>', unsafe_allow_html=True)

with top_cols[1]:
    st.markdown('<div class="dark-toggle-wrapper">', unsafe_allow_html=True)
    toggle_value = st.toggle(
        "Dark mode",
        value=st.session_state.dark_mode,
        key="dark_toggle",
        label_visibility="collapsed",
    )
    st.markdown('<div class="toggle-text toggle-inline">Dark</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    if toggle_value != st.session_state.dark_mode:
        st.session_state.dark_mode = toggle_value
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div id="chat-history" class="chat-history">', unsafe_allow_html=True)
if st.session_state.show_greeting:
    st.markdown(GREETING_HTML, unsafe_allow_html=True)

for idx, message in enumerate(st.session_state.messages):
    safe_content = html.escape(message["content"]).replace("\n", "<br>")
    if message["role"] == "user":
        st.markdown(
            f'<div class="chat-bubble user-bubble">{safe_content}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="chat-bubble bot-bubble">{safe_content}</div>',
            unsafe_allow_html=True,
        )
        play_col, _, _ = st.columns([1, 1, 5])
        with play_col:
            is_playing = idx in st.session_state.audio_playing
            btn_help = "Pause audio" if is_playing else "Play response"
            if st.button("⏯️", key=f"audio_{idx}", help=btn_help):
                if is_playing:
                    try:
                        os.unlink(st.session_state.audio_playing[idx])
                    except OSError:
                        pass
                    del st.session_state.audio_playing[idx]
                else:
                    audio_file = text_to_speech(message["content"], idx)
                    if audio_file:
                        st.session_state.audio_playing[idx] = audio_file
                st.rerun()
        if idx in st.session_state.audio_playing:
            autoplay_audio(st.session_state.audio_playing[idx])

st.markdown("</div>", unsafe_allow_html=True)
st.markdown(
    """
    <script>
        const chatBox = window.parent.document.querySelector('#chat-history');
        if (chatBox) {
            chatBox.scrollTop = chatBox.scrollHeight;
        }
    </script>
    """,
    unsafe_allow_html=True,
)

chips_container = st.container()
with chips_container:
    st.markdown('<div class="prompt-chips">', unsafe_allow_html=True)
    suggestions = get_dynamic_suggestions()
    if suggestions:
        chip_cols = st.columns(len(suggestions))
        for idx, (label, question) in enumerate(suggestions):
            with chip_cols[idx]:
                if st.button(label, key=f"suggestion_{idx}", help=question):
                    st.session_state.text_to_send = question
                    st.session_state.reset_chat_text = True
                    st.rerun()
    else:
        st.caption("Provide any detail to continue your plan.")
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="chat-input">', unsafe_allow_html=True)
input_cols = st.columns([8, 1, 1])

with input_cols[0]:
    st.text_input(
        "Type your message...",
        key="chat_text",
        placeholder="Chat with your AI assistant...",
        label_visibility="collapsed",
        on_change=submit_text_from_box,
    )

with input_cols[1]:
    if st.button("🎤", key="mic_button", help="Speech-to-Text"):
        if not _WHISPER_AVAILABLE:
            st.warning(
                "Speech-to-text requires faster-whisper + torch in this Python environment."
            )
        else:
            voice_text = speech_to_text()
            if voice_text:
                st.session_state.prefill_chat_text = voice_text
                st.session_state.pop("reset_chat_text", None)
                st.rerun()

with input_cols[2]:
    col_send, col_clear = st.columns([2, 1])
    with col_send:
        if st.button("⬆️", key="send_button", help="Send"):
            submit_text_from_box()
    with col_clear:
        if st.button("🗑️", key="clear_chat_footer", help="Clear chat"):
            reset_conversation()
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

pending_text = st.session_state.pop("text_to_send", None)
if pending_text:
    content = pending_text.strip()
    if content:
        st.session_state.messages.append({"role": "user", "content": content})
        process_user_message(content)
        st.session_state.reset_chat_text = True
        st.rerun()
