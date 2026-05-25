import pickle
import librosa
import numpy as np

print("=" * 50)
print("MULTIMODAL EMOTION RECOGNITION")
print("=" * 50)

# Load Fusion Model
with open("results/fusion_model.pkl", "rb") as f:
    fusion_model = pickle.load(f)

# Load Vectorizer
with open("results/fusion_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

audio_path = input("Enter audio path: ")
text_input = input("Enter text: ")

# print("PATH =", audio_path)

audio, sr = librosa.load(audio_path, sr=16000)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

audio_features = np.mean(mfcc, axis=1).reshape(1, -1)

text_features = vectorizer.transform(
    [text_input]
)

from scipy.sparse import hstack

fusion_features = hstack([
    audio_features,
    text_features
])

# print("Audio shape:", audio_features.shape)
# print("Text shape:", text_features.shape)
# print("Fusion shape:", fusion_features.shape)

prediction = fusion_model.predict(
    fusion_features
)

print("\nPredicted Emotion:", prediction[0])