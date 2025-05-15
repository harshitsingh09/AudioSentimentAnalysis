import numpy as np
import librosa
from keras.models import model_from_json
from keras.optimizers import Adam
import pickle

def load_model_and_encoder(model_json_path="model.json", weights_path="Emotion_Model.weights.h5", label_path="labels.pkl"):
    with open(model_json_path, "r") as f:
        model = model_from_json(f.read())
    model.load_weights(weights_path)
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    with open(label_path, "rb") as f:
        lb = pickle.load(f)

    return model, lb

def predict_emotion(audio_path, model, lb, return_probs=False):
    audio, sr = librosa.load(audio_path, sr=44100, duration=2.5, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    mfcc = np.pad(mfcc, ((0, 0), (0, max(0, 216 - mfcc.shape[1]))), mode='constant')[:, :216]
    mfcc = mfcc[np.newaxis, ..., np.newaxis]

    pred = model.predict(mfcc)[0]  # shape: (n_classes,)
    if return_probs:
        return {label: float(prob) for label, prob in zip(lb.classes_, pred)}
    else:
        return lb.inverse_transform([np.argmax(pred)])[0]
