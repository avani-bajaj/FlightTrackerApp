import streamlit as st
import requests
import json
from gtts import gTTS
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'audio_playing' not in st.session_state:
    st.session_state.audio_playing = {}
if 'smart_prompt' not in st.session_state:
    st.session_state.smart_prompt = "SP1"
if 'model' not in st.session_state:
    st.session_state.model = "llama2"

# Load Whisper model once
@st.cache_resource
def load_whisper_model():
    """Load Faster Whisper model"""
    try:
        # Using base model for speed, can use 'small', 'medium', 'large' for better accuracy
        model = WhisperModel("base", device="cpu", compute_type="int8")
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {str(e)}")
        return None

# Custom CSS with Claude/OpenAI style
def get_css(dark_mode):
    if dark_mode:
        return """
        <style>
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            [data-testid="stSidebarNav"] {display: none;}
            
            /* Main background */
            [data-testid="stAppViewContainer"] {
                background-color: #212121;
            }
            
            [data-testid="stHeader"] {
                background-color: transparent;
            }
            
            /* Header */
            .header-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 24px;
                border-bottom: 1px solid #2d2d2d;
                background-color: #212121;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
            }
            
            .header-title {
                font-size: 20px;
                font-weight: 600;
                color: #ececec;
            }
            
            /* Toggle switch */
            .toggle-switch {
                position: relative;
                display: inline-block;
                width: 52px;
                height: 28px;
            }
            
            .toggle-switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #3d3d3d;
                transition: .3s;
                border-radius: 28px;
            }
            
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 20px;
                width: 20px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .3s;
                border-radius: 50%;
            }
            
            input:checked + .toggle-slider {
                background-color: #10a37f;
            }
            
            input:checked + .toggle-slider:before {
                transform: translateX(24px);
            }
            
            /* Main container */
            .main-container {
                padding-top: 72px;
                padding-bottom: 200px;
                min-height: 100vh;
            }
            
            /* Chat messages area */
            .chat-area {
                max-width: 800px;
                margin: 0 auto;
                padding: 24px;
            }
            
            /* Message styles */
            .message-container {
                margin-bottom: 24px;
                display: flex;
                gap: 16px;
                align-items: flex-start;
            }
            
            .message-container.user {
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 4px;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 14px;
            }
            
            .user-avatar {
                background-color: #5436da;
                color: white;
            }
            
            .bot-avatar {
                background-color: #10a37f;
                color: white;
            }
            
            .message-content {
                background-color: transparent;
                color: #ececec;
                padding: 0;
                border-radius: 8px;
                max-width: 70%;
                line-height: 1.7;
                font-size: 15px;
                word-wrap: break-word;
            }
            
            .user-message {
                background-color: #2d2d2d;
                padding: 12px 16px;
                border-radius: 18px;
            }
            
            .bot-message {
                color: #ececec;
            }
            
            /* Audio controls */
            .audio-controls {
                margin-top: 8px;
                display: flex;
                gap: 8px;
            }
            
            /* Smart prompt buttons */
            .prompt-selector {
                max-width: 800px;
                margin: 0 auto 16px;
                display: flex;
                gap: 8px;
                justify-content: center;
                padding: 0 24px;
            }
            
            .prompt-info {
                text-align: center;
                color: #b4b4b4;
                font-size: 12px;
                margin: 8px 0;
            }
            
            /* Input area - fixed at bottom */
            .input-area {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                border-top: 1px solid #2d2d2d;
                padding: 16px 24px 24px;
                background-color: #212121;
                z-index: 1000;
            }
            
            .input-container {
                max-width: 800px;
                margin: 0 auto;
                display: flex;
                gap: 12px;
                align-items: center;
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 12px;
                padding: 4px;
            }
            
            .input-container:focus-within {
                border-color: #565869;
            }
            
            .stTextInput input {
                background-color: transparent !important;
                border: none !important;
                color: #ececec !important;
                font-size: 15px !important;
                padding: 12px !important;
            }
            
            .stTextInput input:focus {
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stTextInput > div > div {
                background-color: transparent !important;
                border: none !important;
            }
            
            .stTextInput > label {
                display: none !important;
            }
            
            /* Buttons */
            .stButton > button {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                color: #b4b4b4;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }
            
            .stButton > button:hover {
                background-color: #3d3d3d;
                color: #ececec;
                border-color: #565869;
            }
            
            .stButton > button:active {
                background-color: #10a37f;
                border-color: #10a37f;
                color: white;
            }
            
            /* Remove Streamlit padding */
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            
            /* Hide sidebar */
            [data-testid="stSidebar"] {
                display: none;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #212121;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #3d3d3d;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #4d4d4d;
            }
        </style>
        """
    else:
        return """
        <style>
            /* Hide Streamlit branding */
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .stDeployButton {display: none;}
            [data-testid="stSidebarNav"] {display: none;}
            
            /* Main background */
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff;
            }
            
            [data-testid="stHeader"] {
                background-color: transparent;
            }
            
            /* Header */
            .header-bar {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 16px 24px;
                border-bottom: 1px solid #e5e5e5;
                background-color: #ffffff;
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                z-index: 1000;
            }
            
            .header-title {
                font-size: 20px;
                font-weight: 600;
                color: #202123;
            }
            
            /* Toggle switch */
            .toggle-switch {
                position: relative;
                display: inline-block;
                width: 52px;
                height: 28px;
            }
            
            .toggle-switch input {
                opacity: 0;
                width: 0;
                height: 0;
            }
            
            .toggle-slider {
                position: absolute;
                cursor: pointer;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background-color: #d1d5db;
                transition: .3s;
                border-radius: 28px;
            }
            
            .toggle-slider:before {
                position: absolute;
                content: "";
                height: 20px;
                width: 20px;
                left: 4px;
                bottom: 4px;
                background-color: white;
                transition: .3s;
                border-radius: 50%;
            }
            
            input:checked + .toggle-slider {
                background-color: #10a37f;
            }
            
            input:checked + .toggle-slider:before {
                transform: translateX(24px);
            }
            
            /* Main container */
            .main-container {
                padding-top: 72px;
                padding-bottom: 200px;
                min-height: 100vh;
            }
            
            /* Chat messages area */
            .chat-area {
                max-width: 800px;
                margin: 0 auto;
                padding: 24px;
            }
            
            /* Message styles */
            .message-container {
                margin-bottom: 24px;
                display: flex;
                gap: 16px;
                align-items: flex-start;
            }
            
            .message-container.user {
                flex-direction: row-reverse;
            }
            
            .message-avatar {
                width: 36px;
                height: 36px;
                border-radius: 4px;
                flex-shrink: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                font-weight: 600;
                font-size: 14px;
            }
            
            .user-avatar {
                background-color: #5436da;
                color: white;
            }
            
            .bot-avatar {
                background-color: #10a37f;
                color: white;
            }
            
            .message-content {
                background-color: transparent;
                color: #374151;
                padding: 0;
                border-radius: 8px;
                max-width: 70%;
                line-height: 1.7;
                font-size: 15px;
                word-wrap: break-word;
            }
            
            .user-message {
                background-color: #f7f7f8;
                padding: 12px 16px;
                border-radius: 18px;
            }
            
            .bot-message {
                color: #374151;
            }
            
            /* Audio controls */
            .audio-controls {
                margin-top: 8px;
                display: flex;
                gap: 8px;
            }
            
            /* Smart prompt buttons */
            .prompt-selector {
                max-width: 800px;
                margin: 0 auto 16px;
                display: flex;
                gap: 8px;
                justify-content: center;
                padding: 0 24px;
            }
            
            .prompt-info {
                text-align: center;
                color: #6b7280;
                font-size: 12px;
                margin: 8px 0;
            }
            
            /* Input area - fixed at bottom */
            .input-area {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                border-top: 1px solid #e5e5e5;
                padding: 16px 24px 24px;
                background-color: #ffffff;
                z-index: 1000;
            }
            
            .input-container {
                max-width: 800px;
                margin: 0 auto;
                display: flex;
                gap: 12px;
                align-items: center;
                background-color: #ffffff;
                border: 1px solid #d1d5db;
                border-radius: 12px;
                padding: 4px;
            }
            
            .input-container:focus-within {
                border-color: #10a37f;
                box-shadow: 0 0 0 1px #10a37f;
            }
            
            .stTextInput input {
                background-color: transparent !important;
                border: none !important;
                color: #374151 !important;
                font-size: 15px !important;
                padding: 12px !important;
            }
            
            .stTextInput input:focus {
                outline: none !important;
                box-shadow: none !important;
            }
            
            .stTextInput > div > div {
                background-color: transparent !important;
                border: none !important;
            }
            
            .stTextInput > label {
                display: none !important;
            }
            
            /* Buttons */
            .stButton > button {
                background-color: #f7f7f8;
                border: 1px solid #d1d5db;
                color: #6b7280;
                padding: 10px 20px;
                border-radius: 8px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 500;
                transition: all 0.2s;
            }
            
            .stButton > button:hover {
                background-color: #e5e7eb;
                color: #374151;
                border-color: #9ca3af;
            }
            
            .stButton > button:active {
                background-color: #10a37f;
                border-color: #10a37f;
                color: white;
            }
            
            /* Remove Streamlit padding */
            .block-container {
                padding: 0 !important;
                max-width: 100% !important;
            }
            
            /* Hide sidebar */
            [data-testid="stSidebar"] {
                display: none;
            }
            
            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 8px;
            }
            
            ::-webkit-scrollbar-track {
                background: #f8f9fa;
            }
            
            ::-webkit-scrollbar-thumb {
                background: #d1d5db;
                border-radius: 4px;
            }
            
            ::-webkit-scrollbar-thumb:hover {
                background: #9ca3af;
            }
        </style>
        """

st.markdown(get_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Smart prompts
SMART_PROMPTS = {
    "SP1": "You are a helpful and friendly AI assistant. Provide clear, concise, and accurate responses.",
    "SP2": "You are an expert AI assistant with deep knowledge. Provide detailed, technical, and comprehensive responses.",
    "ALL": "You are a versatile AI assistant that adapts to user needs. Balance clarity with depth in your responses."
}

DEFAULT_PROMPT = "You are a helpful AI assistant. Provide accurate and friendly responses."

def call_ollama(prompt, user_message, model="llama2"):
    """Call Ollama API with error handling"""
    try:
        url = "http://localhost:11434/api/generate"
        full_prompt = f"{prompt}\n\nUser: {user_message}\nAssistant:"
        payload = {"model": model, "prompt": full_prompt, "stream": False}
        response = requests.post(url, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json().get('response', 'No response generated.')
        else:
            return f"Error: Ollama returned status code {response.status_code}"
    except requests.exceptions.ConnectionError:
        return "Error: Could not connect to Ollama. Please ensure Ollama is running (try 'ollama serve' in terminal)."
    except Exception as e:
        return f"Error: {str(e)}"

def text_to_speech(text, idx):
    """Convert text to speech and return audio file path"""
    try:
        tts = gTTS(text=text, lang='en', slow=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return None

def speech_to_text():
    """Convert speech to text using Faster Whisper"""
    try:
        # Get Whisper model
        whisper_model = load_whisper_model()
        if whisper_model is None:
            st.error("Whisper model not loaded")
            return None
        
        # Recording parameters
        duration = 10  # seconds
        sample_rate = 16000  # Whisper works best with 16kHz
        
        st.info(f"Recording for {duration} seconds... Speak now!")
        
        # Record audio
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        st.success("Processing audio...")
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        wavfile.write(temp_file.name, sample_rate, (audio_data * 32767).astype(np.int16))
        
        # Transcribe with Faster Whisper
        segments, info = whisper_model.transcribe(
            temp_file.name,
            beam_size=5,
            language="en"
        )
        
        # Collect all segments
        transcription = " ".join([segment.text for segment in segments])
        
        # Clean up temp file
        try:
            os.unlink(temp_file.name)
        except:
            pass
        
        if transcription.strip():
            return transcription.strip()
        else:
            st.warning("No speech detected. Please try again.")
            return None
            
    except Exception as e:
        st.error(f"Speech recognition error: {str(e)}")
        return None

# Header with toggle
col1, col2 = st.columns([10, 1])
with col1:
    st.markdown('<div class="header-title">AI Chatbot</div>', unsafe_allow_html=True)
with col2:
    toggle_label = "Dark" if st.session_state.dark_mode else "Light"
    if st.checkbox(toggle_label, value=st.session_state.dark_mode, key="theme_toggle", label_visibility="collapsed"):
        if not st.session_state.dark_mode:
            st.session_state.dark_mode = True
            st.rerun()
    else:
        if st.session_state.dark_mode:
            st.session_state.dark_mode = False
            st.rerun()

# Main container
st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Chat messages area
st.markdown('<div class="chat-area">', unsafe_allow_html=True)

for idx, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]
    
    if role == "user":
        st.markdown(f"""
        <div class="message-container user">
            <div class="message-content">
                <div class="user-message">{content}</div>
            </div>
            <div class="message-avatar user-avatar">U</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="message-container">
            <div class="message-avatar bot-avatar">AI</div>
            <div class="message-content bot-message">{content}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Audio controls
        col1, col2 = st.columns([1, 9])
        with col1:
            if st.button("Play", key=f"play_{idx}"):
                audio_file = text_to_speech(content, idx)
                if audio_file:
                    st.session_state.audio_playing[idx] = audio_file
        with col2:
            if idx in st.session_state.audio_playing:
                if st.button("Stop", key=f"stop_{idx}"):
                    try:
                        os.unlink(st.session_state.audio_playing[idx])
                    except:
                        pass
                    del st.session_state.audio_playing[idx]
                    st.rerun()
        
        if idx in st.session_state.audio_playing:
            st.audio(st.session_state.audio_playing[idx], format='audio/mp3')

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Input area at bottom with smart prompt selector
st.markdown('<div class="input-area">', unsafe_allow_html=True)

# Smart prompt selector buttons
col_sp1, col_sp2, col_sp3, col_sp4 = st.columns([1, 1, 1, 7])
with col_sp1:
    if st.button("SP1", key="sp1_btn", use_container_width=True):
        st.session_state.smart_prompt = "SP1"
with col_sp2:
    if st.button("SP2", key="sp2_btn", use_container_width=True):
        st.session_state.smart_prompt = "SP2"
with col_sp3:
    if st.button("ALL", key="all_btn", use_container_width=True):
        st.session_state.smart_prompt = "ALL"

st.markdown(f"<div class='prompt-info'>Selected: {st.session_state.smart_prompt}</div>", unsafe_allow_html=True)

# Input row
col1, col2, col3 = st.columns([1, 9, 1])

with col1:
    if st.button("Mic", key="mic_btn"):
        voice_text = speech_to_text()
        if voice_text:
            st.session_state.voice_input = voice_text
            st.rerun()

with col2:
    user_input = st.text_input("", key="text_input", placeholder="Message AI Chatbot...")

with col3:
    send_clicked = st.button("Send", key="send_btn")

st.markdown('</div>', unsafe_allow_html=True)

# Process input
process_input = False
current_input = None

if 'voice_input' in st.session_state and st.session_state.voice_input:
    current_input = st.session_state.voice_input
    st.session_state.voice_input = None
    process_input = True
elif send_clicked and user_input:
    current_input = user_input
    process_input = True

if process_input and current_input:
    st.session_state.messages.append({"role": "user", "content": current_input})
    selected_prompt = SMART_PROMPTS.get(st.session_state.smart_prompt, DEFAULT_PROMPT)
    with st.spinner("Thinking..."):
        bot_response = call_ollama(selected_prompt, current_input, st.session_state.model)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    st.rerun()