import streamlit as st
from record import record_audio
from model_utils import load_model_and_encoder, predict_emotion
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf

st.set_page_config(page_title="Live Emotion Detector", layout="centered")

st.title("🎙️ Real-Time Emotion Recognition")
st.markdown("Record your voice and let the model predict your emotion.")

if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model, st.session_state.lb = load_model_and_encoder()

if st.button("🎤 Record Audio"):
    with st.spinner("Recording..."):
        record_audio(filename="test.wav", duration=3)
    st.success("Audio recorded!")

    # Display waveform
    audio, sr = librosa.load("test.wav", sr=None)
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    ax.set_title("Waveform of Your Recording")
    st.pyplot(fig)

    # Predict emotion
    with st.spinner("Predicting..."):
        emotion = predict_emotion("test.wav", st.session_state.model, st.session_state.lb)
    st.success(f"🔊 Predicted Emotion: **{emotion}**")
