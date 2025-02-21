# Required Libraries
import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile
from python_speech_features import mfcc, logfbank
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import pickle
import pyaudio
import wave
import matplotlib.pyplot as plt

# Emotion Mapping
emotions = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}
observed_emotions = ['calm', 'happy', 'fearful', 'disgust']

# Feature Extraction
def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            stft = np.abs(librosa.stft(X))
            chroma_features = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma_features))

        if mel:
            # Corrected line: using keyword arguments
            mel_features = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel_features))

        return result

# Load Dataset and Prepare Data
def load_data(dir_name, test_size=0.25):
    x, y = [], []
    files = glob.glob(os.path.join(dir_name, "Actor_*", "*.wav"))
    print(f"Found {len(files)} .wav files in {dir_name}")

    if len(files) == 0:
        print("No files found. Please check the directory.")
        return [], []

    for file in files:
        file_name = os.path.basename(file)
        emotion = emotions.get(file_name.split("-")[2], None)

        if emotion is None or emotion not in observed_emotions:
            continue
        
        feature = extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)

    if len(x) == 0:
        print("No features extracted. Check if the files follow the correct naming conventions.")
        return [], []

    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)

# Train Model
def train_model(x_train, y_train):
    model = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                          hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(x_train, y_train)
    return model

# Save and Load Model
def save_model(model, file_name="Emotion_Voice_Detection_Model.pkl"):
    with open(file_name, 'wb') as file:
        pickle.dump(model, file)

def load_model(file_name="Emotion_Voice_Detection_Model.pkl"):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

# Real-Time Audio Recording
def record_audio(output_file="output.wav", record_seconds=4, rate=44100, chunk=1024):
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=rate, input=True, frames_per_buffer=chunk)
    print("* Recording...")
    frames = [stream.read(chunk) for _ in range(0, int(rate / chunk * record_seconds))]
    print("* Done Recording")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Real-Time Prediction
def predict_emotion(model, audio_file):
    feature = extract_feature(audio_file, mfcc=True, chroma=True, mel=True)
    feature = feature.reshape(1, -1)
    prediction = model.predict(feature)
    return prediction[0]

# Main Workflow
if __name__ == "__main__":
    # Load and prepare data
    dir_name =r"C:\Hackathon\speech-emotion-recognition-ravdess-data"
    x_train, x_test, y_train, y_test = load_data(dir_name)

    if len(x_train) == 0 or len(x_test) == 0:
        print("No data to split. Please check the file loading issue.")
    else:
        print("Data Loaded:", x_train.shape, x_test.shape)

        # Train the model
        model = train_model(x_train, y_train)
        save_model(model)

        # Test the model
        model = load_model(r"C:\Hackathon\emotion_model.keras")
        print("Model Loaded")
        print("Accuracy:", accuracy_score(y_test, model.predict(x_test)))

        # Real-time recording and prediction
        record_audio(output_file="output.wav", record_seconds=4)
        emotion = predict_emotion(model, "output.wav")
        print("Predicted Emotion:", emotion)