import html
import os
import tempfile
from typing import Optional

import numpy as np
import requests
import scipy.io.wavfile as wavfile
import sounddevice as sd
import streamlit as st
from faster_whisper import WhisperModel
from gtts import gTTS

# ---------------------------------------------------------------------------
# Page + session bootstrap
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "audio_playing" not in st.session_state:
    st.session_state.audio_playing = {}
if "smart_prompt" not in st.session_state:
    st.session_state.smart_prompt = "SP1"
if "model_name" not in st.session_state:
    st.session_state.model_name = "llama2"
if "show_greeting" not in st.session_state:
    st.session_state.show_greeting = True
if "chat_text" not in st.session_state:
    st.session_state.chat_text = ""

# ---------------------------------------------------------------------------
# Prompt presets + helpers
# ---------------------------------------------------------------------------

SMART_PROMPTS = {
    "SP1": "You are a helpful and friendly AI assistant. Provide clear, concise, and accurate responses.",
    "SP2": "You are an expert AI assistant with deep knowledge. Provide detailed, technical, and comprehensive responses.",
    "ALL": "You are a versatile AI assistant that adapts to user needs. Balance clarity with depth in your responses.",
}

DEFAULT_PROMPT = "You are a helpful AI assistant. Provide accurate and friendly responses."
GREETING_HTML = """
<div class="chat-bubble bot-bubble greeting">
    <div class="bubble-title">👋 Welcome to AI Chatbot!</div>
    <div>Hi! I am an AI assistant powered by Ollama open-source models.</div>
    <div>I can help you with questions, conversations, and more!</div>
</div>
"""


@st.cache_resource
def load_whisper_model() -> Optional[WhisperModel]:
    """Load Faster-Whisper once."""
    try:
        return WhisperModel("base", device="cpu", compute_type="int8")
    except Exception as exc:
        st.error(f"Could not load Whisper model: {exc}")
        return None


def call_ollama(prompt: str, user_message: str, model: str = "llama2") -> str:
    """Call Ollama locally with basic error handling."""
    try:
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": model,
            "prompt": f"{prompt}\n\nUser: {user_message}\nAssistant:",
            "stream": False,
        }
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get("response", "No response generated.")
        return f"Error: Ollama returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please ensure Ollama is running (try `ollama serve`)."
    except Exception as exc:
        return f"Error: {exc}"


def text_to_speech(text: str, idx: int) -> Optional[str]:
    """Convert text to speech and return a path."""
    try:
        tts = gTTS(text=text, lang="en", slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as exc:
        st.error(f"Text-to-speech error: {exc}")
        return None


def speech_to_text() -> Optional[str]:
    """Record audio and run it through Faster-Whisper."""
    whisper_model = load_whisper_model()
    if whisper_model is None:
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
        st.success("Audio captured. Transcribing...")

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
    """Remove temp audio files for bot playback."""
    for path in st.session_state.audio_playing.values():
        try:
            os.unlink(path)
        except OSError:
            pass
    st.session_state.audio_playing = {}


def submit_text_from_box():
    """Collect the current text input when ENTER is pressed."""
    text = st.session_state.chat_text.strip()
    if text:
        st.session_state.text_to_send = text


def build_css(dark_mode: bool) -> str:
    palettes = {
        True: {
            "bg": "#05060a",
            "panel": "#11141c",
            "muted": "#8287a2",
            "text": "#f5f6fb",
            "border": "#2c3146",
            "user": "linear-gradient(135deg, #00d2ff 0%, #3a47d5 100%)",
            "bot": "#1c2133",
            "greeting": "#1b243d",
            "input_bg": "#181c27",
            "accent": "#00ffa9",
            "accent_text": "#05060a",
        },
        False: {
            "bg": "#f5f7fb",
            "panel": "#ffffff",
            "muted": "#6f7485",
            "text": "#111322",
            "border": "#d9dbe0",
            "user": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "bot": "#f2f4ff",
            "greeting": "#e6ecff",
            "input_bg": "#f1f3f7",
            "accent": "#4f46e5",
            "accent_text": "#ffffff",
        },
    }[dark_mode]

    return f"""
    <style>
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
        [data-testid="stSidebar"], footer, header {{
            display: none;
        }}
        .main .block-container {{
            padding: 1.25rem 1.5rem 1.5rem;
            height: 100vh;
            box-sizing: border-box;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
        }}
        .hero-title {{
            font-size: 1.9rem;
            font-weight: 700;
        }}
        .hero-sub {{
            color: {palettes["muted"]};
            font-size: 0.95rem;
        }}
        .chat-shell {{
            background: {palettes["panel"]};
            border: 1px solid {palettes["border"]};
            border-radius: 24px;
            box-shadow: 0 15px 35px rgba(0,0,0,0.25);
            padding: 1.5rem;
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 1rem;
            min-height: 400px;
        }}
        .chat-history {{
            flex: 1;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            justify-content: flex-end;
            gap: 0.75rem;
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
            padding: 0.9rem 1.1rem;
            border-radius: 20px;
            font-size: 0.98rem;
            line-height: 1.4;
            max-width: 85%;
        }}
        .user-bubble {{
            background: {palettes["user"]};
            color: #ffffff;
            margin-left: auto;
            border-bottom-right-radius: 6px;
        }}
        .bot-bubble {{
            background: {palettes["bot"]};
            color: {palettes["text"]};
            margin-right: auto;
            border-bottom-left-radius: 6px;
        }}
        .greeting {{
            background: {palettes["greeting"]};
            border: 1px solid {palettes["border"]};
        }}
        .bubble-title {{
            font-weight: 700;
            margin-bottom: 0.35rem;
        }}
        .chat-input {{
            display: flex;
            align-items: center;
            gap: 0.75rem;
            border-top: 1px solid {palettes["border"]};
            padding-top: 0.75rem;
        }}
        .chat-input .stTextInput > div > div {{
            border-radius: 999px;
            border: 1px solid {palettes["border"]};
            background: {palettes["input_bg"]};
        }}
        .chat-input .stTextInput > div > div > input {{
            padding: 0.6rem 1.2rem;
            font-size: 1rem;
            color: {palettes["text"]};
            background: transparent;
        }}
        .chat-input .stButton button {{
            border-radius: 999px;
            height: 52px;
            width: 52px;
            background: {palettes["accent"]};
            color: {palettes["accent_text"]};
            border: none;
            font-size: 1.4rem;
        }}
        .chat-input .stButton button:hover {{
            filter: brightness(1.1);
        }}
        .prompt-row {{
            margin-top: 0.25rem;
        }}
        div[data-testid="stHorizontalBlock"] label {{
            background: {palettes["panel"]};
            border: 1px solid {palettes["border"]};
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            margin-right: 0.5rem;
            color: {palettes["muted"]};
            font-weight: 600;
        }}
        div[data-testid="stHorizontalBlock"] label input {{
            display: none;
        }}
        div[data-testid="stHorizontalBlock"] label[data-selected="true"] {{
            background: {palettes["accent"]};
            color: {palettes["accent_text"]};
            border-color: {palettes["accent"]};
        }}
        .model-input .stTextInput > div > div {{
            border-radius: 14px;
            border: 1px solid {palettes["border"]};
            background: {palettes["panel"]};
        }}
        .model-input input {{
            font-weight: 600;
        }}
    </style>
    """


st.markdown(build_css(st.session_state.dark_mode), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header + controls
# ---------------------------------------------------------------------------

title_col, toggle_col = st.columns([7, 1])
with title_col:
    st.markdown(
        '<div class="hero-title">🤖 AI Chatbot</div>'
        '<div class="hero-sub">Powered by Ollama open-source models</div>',
        unsafe_allow_html=True,
    )
with toggle_col:
    toggle_value = st.toggle("Dark mode", value=st.session_state.dark_mode)
    if toggle_value != st.session_state.dark_mode:
        st.session_state.dark_mode = toggle_value
        st.rerun()

model_col, clear_col = st.columns([6, 1])
with model_col:
    with st.container():
        st.markdown('<div class="model-input">', unsafe_allow_html=True)
        model_value = st.text_input(
            "Model",
            value=st.session_state.model_name,
            placeholder="Ollama model (e.g., llama3)",
            label_visibility="collapsed",
            key="model_input",
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.model_name = model_value.strip() or st.session_state.model_name
with clear_col:
    if st.button("🧹", help="Clear conversation", use_container_width=True):
        clear_audio_buffers()
        st.session_state.messages = []
        st.session_state.show_greeting = True
        st.session_state.chat_text = ""
        st.session_state.pop("text_to_send", None)
        st.rerun()

prompt_choice = st.radio(
    "Prompt Style",
    ["SP1", "SP2", "ALL"],
    index=["SP1", "SP2", "ALL"].index(st.session_state.smart_prompt),
    horizontal=True,
    label_visibility="collapsed",
)
if prompt_choice != st.session_state.smart_prompt:
    st.session_state.smart_prompt = prompt_choice

# ---------------------------------------------------------------------------
# Chat surface
# ---------------------------------------------------------------------------

st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

st.markdown('<div class="chat-history">', unsafe_allow_html=True)
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
        play_col, stop_col, _ = st.columns([1, 1, 6])
        with play_col:
            if st.button("▶️", key=f"play_{idx}", help="Play response"):
                audio_file = text_to_speech(message["content"], idx)
                if audio_file:
                    st.session_state.audio_playing[idx] = audio_file
        with stop_col:
            if idx in st.session_state.audio_playing:
                if st.button("⏹️", key=f"stop_{idx}", help="Stop audio"):
                    try:
                        os.unlink(st.session_state.audio_playing[idx])
                    except OSError:
                        pass
                    del st.session_state.audio_playing[idx]
                    st.rerun()
        if idx in st.session_state.audio_playing:
            st.audio(st.session_state.audio_playing[idx], format="audio/mp3")

st.markdown("</div>", unsafe_allow_html=True)

st.markdown('<div class="chat-input">', unsafe_allow_html=True)
input_cols = st.columns([9, 1, 1])

with input_cols[0]:
    st.text_input(
        "Your message",
        key="chat_text",
        placeholder="Type your message...",
        label_visibility="collapsed",
        on_change=submit_text_from_box,
    )

with input_cols[1]:
    if st.button("🎤", key="mic_icon", help="Speak instead of typing"):
        voice_text = speech_to_text()
        if voice_text:
            st.session_state.text_to_send = voice_text
            st.session_state.chat_text = ""

with input_cols[2]:
    if st.button("⬆️", key="send_icon", help="Send message"):
        submit_text_from_box()

st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.caption("🚀 Make sure Ollama is running locally (`ollama serve`).")

# ---------------------------------------------------------------------------
# Message processing
# ---------------------------------------------------------------------------

pending_text = st.session_state.pop("text_to_send", None)
if pending_text:
    content = pending_text.strip()
    if content:
        st.session_state.show_greeting = False
        st.session_state.messages.append({"role": "user", "content": content})
        selected_prompt = SMART_PROMPTS.get(
            st.session_state.smart_prompt, DEFAULT_PROMPT
        )
        with st.spinner("🤔 Thinking..."):
            reply = call_ollama(selected_prompt, content, st.session_state.model_name)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.session_state.chat_text = ""
        st.rerun()
