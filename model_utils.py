import numpy as np
import librosa
from keras.models import model_from_json
from keras.optimizers import Adam
import pickle

def load_model_and_encoder(model_json_path='model.json',
                           weights_path='Emotion_Model.weights.h5',
                           label_path='labels.pkl'):
    """
    Loads the trained model and label encoder.

    Returns:
        model (keras.Model): Loaded Keras model
        lb (LabelEncoder): Loaded label encoder
    """
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(weights_path)

    opt = Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    with open(label_path, 'rb') as f:
        lb = pickle.load(f)

    return model, lb


def extract_features(audio_path, n_mfcc=30):
    """
    Extracts MFCC features from the audio file.

    Returns:
        np.ndarray: Preprocessed MFCC input for model (shape: 1, 30, 216, 1)
    """
    data, sr = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=44100, offset=0.5)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)

    # Ensure consistent shape (30, 216)
    if mfcc.shape[1] < 216:
        pad_width = 216 - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :216]

    mfcc = np.expand_dims(mfcc, axis=-1)  # (30, 216, 1)
    mfcc = np.expand_dims(mfcc, axis=0)   # (1, 30, 216, 1)
    return mfcc


def predict_emotion(audio_path, model, lb):
    """
    Predicts the emotion from an audio file.

    Returns:
        str: Predicted emotion label
    """
    features = extract_features(audio_path)
    prediction = model.predict(features)
    label_idx = prediction.argmax(axis=1)
    label = lb.inverse_transform(label_idx)
    return label[0]
