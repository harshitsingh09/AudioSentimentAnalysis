import streamlit as st
from record import record_audio
from model_utils import load_model_and_encoder, predict_emotion
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np

st.set_page_config(page_title="Live Emotion Detector", layout="centered")
st.title("🎙️ Real-Time Emotion Recognition")
st.markdown("Record your voice and let the model predict your emotion.")

# Sidebar info
st.sidebar.header("ℹ️ About This App")
st.sidebar.markdown("""
This app records your voice and predicts the emotion behind it using a trained neural network model.

**Supported Emotions**: Happy, Sad, Angry, Neutral, Fearful, Disgust, Surprised, Calm  
**Model**: CNN trained on MFCC features from RAVDESS dataset.
""")

# Cache model loading
@st.cache_resource
def load_model():
    return load_model_and_encoder()

# Load model once
if 'model' not in st.session_state:
    with st.spinner("Loading model..."):
        st.session_state.model, st.session_state.lb = load_model()

# Recording duration control
duration = st.slider("Select Recording Duration (seconds)", 2, 10, 3)

if st.button("🎤 Record Audio"):
    try:
        with st.spinner("Recording..."):
            record_audio(filename="test.wav", duration=duration)
        st.success("Audio recorded!")

        # Display waveform
        audio, sr = librosa.load("test.wav", sr=None)
        fig, ax = plt.subplots()
        librosa.display.waveshow(audio, sr=sr, ax=ax)
        ax.set_title("Waveform of Your Recording")
        st.pyplot(fig)

        # Audio playback
        st.audio("test.wav", format="audio/wav")

        # Predict emotion
        with st.spinner("Predicting..."):
            probs = predict_emotion("test.wav", st.session_state.model, st.session_state.lb, return_probs=True)

        # Sort and display top-3
        top3 = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
        top_emotion, confidence = top3[0]
        st.success(f"🔊 Predicted Emotion: **{top_emotion}** ({confidence*100:.1f}%)")

        st.markdown("### 🔍 Top Predictions:")
        for label, prob in top3:
            st.progress(min(prob, 1.0), text=f"{label}: {prob*100:.1f}%")

    except Exception as e:
        st.error(f"Something went wrong: {str(e)}")
