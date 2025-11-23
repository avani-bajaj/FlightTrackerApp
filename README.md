# AirScout – Flight Tracker App

AirScout is a conversational US flight–planning assistant built with Streamlit.  
It gathers your origin, destination, budget, dates, and stop preferences, then searches the Amadeus Flight Offers API for itineraries that match. The assistant can clarify missing details, restyle results via a local Ollama model, and even speak and listen thanks to gTTS + Faster-Whisper.

## Features
- **Guided slot-filling chat** – Streamlit UI with dynamic follow-ups to capture origin, destination, travel window, trip length, budget, and max stops.
- **Real flight data** – Queries the Amadeus Self-Service APIs (test/sandbox) once the traveler confirms the trip brief.
- **LLM tone + extraction** – Uses Ollama locally to rephrase responses and help interpret free-form text.
- **Voice I/O** – Optional microphone capture + Faster-Whisper transcription and gTTS playback for each assistant reply.
- **Rich UI polish** – Custom CSS, dark/light toggle, suggestion chips, and audio controls for every message bubble.
- **Archived prototypes** – Earlier experiments live in `Appendix/` for reference.

## Project Layout
- `app.py` – Streamlit application that powers the AirScout chat experience (entry point for Streamlit Community Cloud).
- `Appendix/` – Earlier iterations (`app_avani_v1_*`, `app_final.py`, etc.) kept for reference.
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
- **Speech-to-text (optional)**: Install the `sounddevice` package manually (`pip install sounddevice`) and ensure PortAudio libs exist (e.g., `brew install portaudio`). This stack is omitted from Streamlit Cloud builds, so the microphone toggle stays disabled there.
- **Text-to-speech**: gTTS needs outbound network once to hit Google’s TTS endpoint.
- **Ollama**: Install Ollama locally and pull the configured model (`ollama pull gemma3:1b`) or update `OLLAMA_MODEL` to one you already have.

#### Ollama Setup
1. Install Ollama from https://ollama.com/download (macOS/Linux) and ensure the `ollama` CLI is on your `PATH`.
2. Start the Ollama background service (`ollama serve`) if it isn’t already running.
3. Pull the model used by AirScout:
   ```bash
   ollama pull gemma3:1b
   ```
4. Optionally, set `OLLAMA_URL` and `OLLAMA_MODEL` in your environment if you’re hosting Ollama remotely or prefer a different model.
5. Run `ollama list` to confirm the model is available before launching Streamlit.

## Deploying to Streamlit Community Cloud
1. Push changes to GitHub (already set up for `avani-bajaj/FlightTrackerApp`).
2. In Streamlit Cloud, choose “New app” → select the repo and `main` branch → set the entry point to `app.py`.
3. Configure `AMADEUS_CLIENT_ID`, `AMADEUS_CLIENT_SECRET`, `OLLAMA_URL`, and `OLLAMA_MODEL` as **Secrets** in the Streamlit Cloud workspace.
4. Deploy. Streamlit Cloud automatically installs `requirements.txt` and runs the entrypoint on each push.

## Troubleshooting
- **Amadeus auth errors** – Confirm credentials and that you are still using the test environment (calls are limited).
- **Audio capture failures** – Ensure your OS grants microphone access and that PortAudio/sounddevice detect an input device.
- **Ollama timeouts** – If the Ollama host isn’t reachable/slow, AirScout falls back to structured summaries automatically.

Feel free to file issues or submit PRs with improvements, new cities, or better UI polish. Happy flying! ✈️
