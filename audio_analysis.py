import os
import librosa
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the model and label encoder when the module is imported
model_path = "best_xgb_model.pkl"  # Make sure this path is correct
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

le = LabelEncoder()
emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
le.fit(emotions)

def extract_features(X, sr):
    features = []

    # MFCCs
    mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=60)
    features.extend(np.mean(mfccs, axis=1))
    features.extend(np.std(mfccs, axis=1))

    # Spectral Centroid
    spectral_centroids = librosa.feature.spectral_centroid(y=X, sr=sr)[0]
    features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])

    # Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=X, sr=sr)[0]
    features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])

    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(X)[0]
    features.extend([np.mean(zcr), np.std(zcr)])

    # Chroma Features
    chroma = librosa.feature.chroma_stft(y=X, sr=sr)
    features.extend(np.mean(chroma, axis=1))

    return np.array(features)

def recognize_emotion(audio_file):
    print(f"Analyzing file: {audio_file}")
    audio, sr = librosa.load(audio_file, duration=3)  # Load only first 3 seconds
    features = extract_features(audio, sr).reshape(1, -1)
    print(f"Extracted {features.shape[1]} features")
    probabilities = model.predict_proba(features)[0]
    emotion_scores = {le.inverse_transform([i])[0]: prob for i, prob in enumerate(probabilities)}
    return emotion_scores

if __name__ == "__main__":
    while True:
        audio_file = input("Enter the path to the audio file (or 'q' to quit): ").strip()

        if audio_file.lower() == 'q':
            break

        if not os.path.exists(audio_file):
            print(f"File not found: {audio_file}")
            continue

        try:
            emotion_scores = recognize_emotion(audio_file)

            print("\nEmotion Scores:")
            for emotion, score in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True):
                print(f"{emotion}: {score:.4f}")

            top_emotion = max(emotion_scores, key=emotion_scores.get)
            print(f"\nTop predicted emotion: {top_emotion}")
        except Exception as e:
            print(f"An error occurred: {str(e)}")

        print("\n" + "-" * 50 + "\n")

    print("Thank you for using the Audio Emotion Recognition Tester!")