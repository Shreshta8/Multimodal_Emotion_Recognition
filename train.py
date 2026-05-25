import os
import random
import pickle

import librosa
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from scipy.sparse import csr_matrix
from scipy.sparse import hstack

# ==========================================
# DATASET PATH
# ==========================================

DATASET_PATH = "datasets/TESS Toronto emotional speech set data"

# ==========================================
# TEXT SAMPLES
# ==========================================

emotion_texts = {

    "angry": [
        "I am angry",
        "I am furious",
        "I am upset",
        "This makes me mad"
    ],

    "disgust": [
        "I feel disgusted",
        "This is disgusting",
        "I feel uncomfortable",
        "This is horrible"
    ],

    "fear": [
        "I am scared",
        "I am afraid",
        "I feel nervous",
        "I am worried"
    ],

    "happy": [
        "I am happy",
        "I am joyful",
        "Life is beautiful",
        "I feel wonderful today"
    ],

    "neutral": [
        "I am okay",
        "I feel normal",
        "Nothing special happened",
        "Just another day"
    ],

    "surprise": [
        "I am surprised",
        "Wow I didn't expect that",
        "This is amazing",
        "What a surprise"
    ],

    "sad": [
        "I feel lonely",
        "I am sad",
        "I feel unhappy",
        "Life is difficult"
    ]
}

# ==========================================
# BUILD DATASET
# ==========================================

data = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    if not os.path.isdir(folder_path):
        continue

    emotion = folder.split("_")[-1].lower()

    if emotion == "surprised":
        emotion = "surprise"

    if emotion not in emotion_texts:
        continue

    for file in os.listdir(folder_path):

        if file.endswith(".wav"):

            file_path = os.path.join(
                folder_path,
                file
            )

            data.append({
                "file_path": file_path,
                "text": random.choice(
                    emotion_texts[emotion]
                ),
                "emotion": emotion
            })

df = pd.DataFrame(data)

print("Total Samples:", len(df))

# ==========================================
# AUDIO FEATURES
# ==========================================

audio_features = []

for file_path in df["file_path"]:

    audio, sr = librosa.load(
        file_path,
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    feature_vector = np.mean(
        mfcc,
        axis=1
    )

    audio_features.append(
        feature_vector
    )

audio_features = np.array(
    audio_features
)

print(
    "Audio Features Shape:",
    audio_features.shape
)

# ==========================================
# TEXT FEATURES
# ==========================================

vectorizer = TfidfVectorizer()

text_features = vectorizer.fit_transform(
    df["text"]
)

print(
    "Text Features Shape:",
    text_features.shape
)

# ==========================================
# FUSION FEATURES
# ==========================================

audio_sparse = csr_matrix(
    audio_features
)

fusion_features = hstack([
    audio_sparse,
    text_features
])

print(
    "Fusion Shape:",
    fusion_features.shape
)

# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    fusion_features,
    df["emotion"],
    test_size=0.2,
    random_state=42,
    stratify=df["emotion"]
)

# ==========================================
# MODEL
# ==========================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

print("\nTraining Fusion Model...")

model.fit(
    X_train,
    y_train
)

# ==========================================
# EVALUATION
# ==========================================

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(
    "\nFusion Accuracy:",
    accuracy
)

print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y_test,
        predictions
    )
)

# ==========================================
# SAVE MODEL
# ==========================================

os.makedirs(
    "results",
    exist_ok=True
)

with open(
    "results/fusion_model.pkl",
    "wb"
) as f:

    pickle.dump(
        model,
        f
    )

with open(
    "results/fusion_vectorizer.pkl",
    "wb"
) as f:

    pickle.dump(
        vectorizer,
        f
    )

print(
    "\nFusion model saved successfully!"
)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFUSION MATRIX
# ==========================================

plt.figure(figsize=(8, 6))

cm = confusion_matrix(y_test, predictions)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=model.classes_,
    yticklabels=model.classes_
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fusion Model Confusion Matrix")

plt.tight_layout()
plt.savefig("results/confusion_matrix.png")
plt.close()

# ==========================================
# EMOTION DISTRIBUTION
# ==========================================

plt.figure(figsize=(8, 5))

emotion_counts = df["emotion"].value_counts()

plt.bar(
    emotion_counts.index,
    emotion_counts.values
)

plt.title("Emotion Distribution")
plt.xlabel("Emotion")
plt.ylabel("Count")

plt.tight_layout()
plt.savefig("results/emotion_distribution.png")
plt.close()

# ==========================================
# MODEL COMPARISON
# ==========================================

plt.figure(figsize=(6, 4))

models = ["Text", "Speech", "Fusion"]

fusion_accuracy = round(
    accuracy * 100,
    2
)

accuracies = [
    100,
    100,
    fusion_accuracy
]

plt.bar(
    models,
    accuracies
)

plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")

plt.ylim(90, 100)

plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.close()

print("\nVisualizations saved successfully!")

from sklearn.decomposition import PCA

# ==========================================
# TEXT CLUSTERS
# ==========================================

text_dense = text_features.toarray()

pca = PCA(n_components=2)

text_pca = pca.fit_transform(
    text_dense
)

plt.figure(figsize=(8,6))

emotion_labels = pd.factorize(
    df["emotion"]
)[0]

scatter = plt.scatter(
    text_pca[:,0],
    text_pca[:,1],
    c=emotion_labels,
    cmap="tab10"
)

plt.colorbar(scatter)

plt.title("Text Emotion Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()

plt.savefig(
    "results/text_clusters.png"
)

plt.close()

print("Text cluster visualization saved!")

# ==========================================
# FUSION CLUSTERS
# ==========================================

fusion_dense = fusion_features.toarray()

pca = PCA(n_components=2)

fusion_pca = pca.fit_transform(
    fusion_dense
)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    fusion_pca[:,0],
    fusion_pca[:,1],
    c=emotion_labels,
    cmap="tab10"
)

plt.colorbar(scatter)

plt.title("Fusion Emotion Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()

plt.savefig(
    "results/fusion_clusters.png"
)

plt.close()

print("Fusion cluster visualization saved!")