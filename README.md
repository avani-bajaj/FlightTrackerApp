# AirScout – Flight Tracker App

AirScout is a conversational US flight–planning assistant built with Streamlit.  
It gathers your origin, destination, budget, dates, and stop preferences, then searches the Amadeus Flight Offers API for itineraries that match. The assistant can clarify missing details, restyle results via a local Ollama model, and even speak and listen thanks to gTTS + Faster-Whisper.

## Features
- **Guided slot-filling chat** – Streamlit UI with dynamic follow-ups to capture origin, destination, travel window, trip length, budget, and max stops.
- **Real flight data** – Queries the Amadeus Self-Service APIs (test/sandbox) once the traveler confirms the trip brief.
- **LLM tone + extraction** – Uses Ollama locally to rephrase responses and help interpret free-form text.
- **Voice I/O** – Optional microphone capture + Faster-Whisper transcription and gTTS playback for each assistant reply.
- **Rich UI polish** – Custom CSS, dark/light toggle, suggestion chips, and audio controls for every message bubble.
- **Flask fallback** – A lighter web UI lives in `app.py` if you need a classic server-rendered experience.

## Project Layout
- `app.py` – Streamlit application that powers the AirScout chat experience (entry point for Streamlit Community Cloud).
- `Appendix/app.py` – Original Flask prototype with similar Amadeus slot-filling logic.
- `Appendix/` – Earlier Streamlit/Flask iterations (`app_avani_v1_*`, `app_final.py`, etc.) kept for reference.
- `requirements.txt` – Python dependencies for Streamlit Cloud or local installs.
- `static/` – Placeholder assets.

## Getting Started Locally
1. **Clone the repo**  
   ```bash
   git clone https://github.com/avani-bajaj/FlightTrackerApp.git
   cd FlightTrackerApp
   ```
2. **Install dependencies**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Set required environment variables**  
   ```bash
   export AMADEUS_CLIENT_ID="your_client_id"
   export AMADEUS_CLIENT_SECRET="your_client_secret"
   # Optional: point AirScout at a custom Ollama instance/model
   export OLLAMA_URL="http://127.0.0.1:11434"
   export OLLAMA_MODEL="gemma3:1b"
   ```
4. **Run Streamlit**  
   ```bash
   streamlit run app.py
   ```
   The app launches at `http://localhost:8501`.

### Audio/LLM Prerequisites
- **Speech-to-text**: Requires a working microphone plus `faster-whisper` (already listed) and its dependencies. On macOS, install PortAudio for PyAudio: `brew install portaudio`.
- **Text-to-speech**: gTTS needs outbound network once to hit Google’s TTS endpoint.
- **Ollama**: Install Ollama locally and pull the configured model (`ollama pull gemma3:1b`) or update `OLLAMA_MODEL` to one you already have.

## Deploying to Streamlit Community Cloud
1. Push changes to GitHub (already set up for `avani-bajaj/FlightTrackerApp`).
2. In Streamlit Cloud, choose “New app” → select the repo and `main` branch → set the entry point to `app.py`.
3. Configure `AMADEUS_CLIENT_ID`, `AMADEUS_CLIENT_SECRET`, `OLLAMA_URL`, and `OLLAMA_MODEL` as **Secrets** in the Streamlit Cloud workspace.
4. Deploy. Streamlit Cloud automatically installs `requirements.txt` and runs the entrypoint on each push.

## Running the Flask Prototype
```bash
export AMADEUS_CLIENT_ID=...
export AMADEUS_CLIENT_SECRET=...
flask --app Appendix.app run
```
This serves a lighter HTML chatbot (no Streamlit UI). Use it if you prefer classic server-side rendering.

## Troubleshooting
- **Amadeus auth errors** – Confirm credentials and that you are still using the test environment (calls are limited).
- **Audio capture failures** – Ensure PyAudio detected your input device; on macOS allow microphone access in System Settings.
- **Ollama timeouts** – If the Ollama host isn’t reachable/slow, AirScout falls back to structured summaries automatically.

Feel free to file issues or submit PRs with improvements, new cities, or better UI polish. Happy flying! ✈️
