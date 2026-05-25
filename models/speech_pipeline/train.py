import os
import pandas as pd
import librosa
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split


# =========================
# LOAD DATASET
# =========================

DATASET_PATH = "datasets/TESS Toronto emotional speech set data"

data = []

for folder in os.listdir(DATASET_PATH):

    folder_path = os.path.join(DATASET_PATH, folder)

    if os.path.isdir(folder_path):

        emotion = folder.split("_")[-1].lower()

        for file in os.listdir(folder_path):

            if file.endswith(".wav"):

                file_path = os.path.join(folder_path, file)

                data.append({
                    "path": file_path,
                    "emotion": emotion
                })

df = pd.DataFrame(data)


# =========================
# EMOTION MAPPING
# =========================

emotion_map = {
    "angry": 0,
    "disgust": 1,
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "surprise": 5,
    "surprised": 5,
    "sad": 6
}

print(df["emotion"].unique())


# =========================
# MFCC EXTRACTION
# =========================

X = []
y = []

for index, row in df.iterrows():

    audio, sr = librosa.load(
        row["path"],
        sr=16000
    )

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    if mfcc.shape[1] < 100:

        pad_width = 100 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:
        mfcc = mfcc[:, :100]

    X.append(mfcc)

    emotion = row["emotion"]

    y.append(emotion_map[emotion])


X = np.array(X)
y = np.array(y)

print("X Shape:", X.shape)
print("y Shape:", y.shape)


# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# =========================
# TENSOR CONVERSION
# =========================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

X_test = torch.tensor(
    X_test,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.long
)

y_test = torch.tensor(
    y_test,
    dtype=torch.long
)


# =========================
# DATALOADERS
# =========================

train_dataset = TensorDataset(
    X_train,
    y_train
)

test_dataset = TensorDataset(
    X_test,
    y_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

for X_batch, y_batch in train_loader:
    print(X_batch.shape)
    print(y_batch.shape)
    break


# =========================
# CNN MODEL
# =========================

class SpeechCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=40,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.pool = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.fc1 = nn.Linear(
            128 * 25,
            128
        )

        self.fc2 = nn.Linear(
            128,
            7
        )

        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(
            self.conv1(x)
        )

        x = self.pool(x)

        x = self.relu(
            self.conv2(x)
        )

        x = self.pool(x)

        x = x.view(
            x.size(0),
            -1
        )

        x = self.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x


# =========================
# MODEL
# =========================

model = SpeechCNN()

print(model)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("Loss and Optimizer Created")


# =========================
# FORWARD PASS TEST
# =========================

for X_batch, y_batch in train_loader:

    outputs = model(X_batch)

    print(
        "Output Shape:",
        outputs.shape
    )

    break


epochs = 10

for epoch in range(epochs):

    model.train()

    running_loss = 0

    for X_batch, y_batch in train_loader:

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(outputs, y_batch)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print(
        f"Epoch [{epoch+1}/{epochs}] Loss: {running_loss/len(train_loader):.4f}"
    )

# ==========================
# TEST ACCURACY
# ==========================

correct = 0
total = 0

model.eval()

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        _, predicted = torch.max(outputs, 1)

        total += y_batch.size(0)

        correct += (predicted == y_batch).sum().item()

accuracy = 100 * correct / total

print(f"\nTest Accuracy: {accuracy:.2f}%")

torch.save(
    model.state_dict(),
    "results/speech_model.pth"
)

print("Speech model saved!")

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ==========================================
# SPEECH CLUSTERS
# ==========================================

pca = PCA(n_components=2)

X_flat = X.reshape(X.shape[0], -1)

speech_pca = pca.fit_transform(X_flat)

plt.figure(figsize=(8,6))

scatter = plt.scatter(
    speech_pca[:,0],
    speech_pca[:,1],
    c=y,
    cmap="tab10"
)

plt.colorbar(scatter)

plt.title("Speech Emotion Clusters")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")

plt.tight_layout()

plt.savefig("results/speech_clusters.png")

plt.close()

print("Speech cluster visualization saved!")