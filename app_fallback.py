import streamlit as st
from record import record_audio
from transcribe import transcribe_audio
from sentiment import analyze_sentiment
import librosa
import librosa.display
import matplotlib.pyplot as plt

st.set_page_config(page_title="Whisper Sentiment Detector", layout="centered")

st.title("📝 Speech Transcription & Sentiment Analysis")
st.markdown("Record your voice. Get a text transcript + sentiment analysis.")

# Sidebar Info
st.sidebar.markdown("""
### ℹ️ What this app does:
- Records your voice
- Transcribes it using OpenAI's **Whisper**
- Analyzes the **sentiment** of what you said using a pretrained model

No custom ML models involved.
""")

duration = st.slider("Recording duration (seconds)", 2, 10, 3)

if st.button("🎤 Record & Analyze"):
    try:
        with st.spinner("Recording..."):
            record_audio("test.wav", duration)
        st.success("Recording complete!")

        st.audio("test.wav", format="audio/wav")

        # Waveform
        audio, sr = librosa.load("test.wav", sr=None)
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title("Waveform")
        st.pyplot(fig)

        # Transcription
        with st.spinner("Transcribing..."):
            transcript = transcribe_audio("test.wav")

        st.markdown("### 📝 Transcription")
        st.info(transcript if transcript.strip() else "_No speech detected_")

        # Sentiment
        if transcript.strip():
            label, score = analyze_sentiment(transcript)
            st.markdown(f"### 🗣️ Sentiment: **{label}** ({score*100:.1f}%)")
        else:
            st.warning("Nothing to analyze.")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
