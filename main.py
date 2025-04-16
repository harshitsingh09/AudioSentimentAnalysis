from record import record_audio
from model_utils import load_model_and_encoder, predict_emotion
import os

def main():
    audio_file = "test.wav"
    
    # Record new audio
    record_audio(filename=audio_file, duration=3)

    # Load model + label encoder
    model_path = "model.json"
    weights_path = "Emotion_Model.weights.h5"
    label_path = "labels.pkl"

    if not all(os.path.exists(p) for p in [model_path, weights_path, label_path]):
        print("[ERROR] Model files not found. Please ensure 'model.json', 'Emotion_Model.weights.h5', and 'labels.pkl' are in the project folder.")
        return

    model, lb = load_model_and_encoder(
        model_json_path=model_path,
        weights_path=weights_path,
        label_path=label_path
    )

    # Predict emotion from the recorded audio
    try:
        emotion = predict_emotion(audio_file, model, lb)
        print(f"[RESULT] Detected Emotion: {emotion}")
    except Exception as e:
        print(f"[ERROR] Failed to predict emotion: {str(e)}")

if __name__ == "__main__":
    main()
