import pickle
import librosa
import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

# ==========================
# LOAD MODELS
# ==========================

with open("results/fusion_model.pkl", "rb") as f:
    fusion_model = pickle.load(f)

with open("results/fusion_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# ==========================
# INPUTS
# ==========================

audio_path = input("Enter audio path: ")

text_input = input("Enter text: ")

# ==========================
# AUDIO FEATURES
# ==========================

audio, sr = librosa.load(
    audio_path,
    sr=16000
)

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

audio_features = np.mean(
    mfcc,
    axis=1
)

audio_features = audio_features.reshape(1, -1)

audio_sparse = csr_matrix(audio_features)

# ==========================
# TEXT FEATURES
# ==========================

text_features = vectorizer.transform(
    [text_input]
)

# ==========================
# FUSION
# ==========================

fusion_features = hstack([
    audio_sparse,
    text_features
])

prediction = fusion_model.predict(
    fusion_features
)

print("\nPredicted Emotion:", prediction[0])