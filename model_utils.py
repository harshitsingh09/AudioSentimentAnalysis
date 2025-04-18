import numpy as np
import librosa
from keras.models import model_from_json
from keras.optimizers import Adam
import pickle

def load_model_and_encoder():
    with open("model.json", "r") as f:
        model = model_from_json(f.read())
    model.load_weights("Emotion_Model.weights.h5")
    model.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    with open("labels.pkl", "rb") as f:
        lb = pickle.load(f)

    return model, lb

def predict_emotion(audio_path, model, lb):
    audio, sr = librosa.load(audio_path, sr=44100, duration=2.5, offset=0.5)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=30)
    if mfcc.shape[1] < 216:
        pad_width = 216 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0,0), (0,pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :216]
    mfcc = mfcc[np.newaxis, ..., np.newaxis]  # shape: (1, 30, 216, 1)

    pred = model.predict(mfcc)
    label_idx = np.argmax(pred, axis=1)
    return lb.inverse_transform(label_idx)[0]
