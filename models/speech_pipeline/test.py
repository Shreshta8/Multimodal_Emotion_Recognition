import torch
import torch.nn as nn
import librosa
import numpy as np


# ==========================
# CNN MODEL
# ==========================

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


# ==========================
# LOAD MODEL
# ==========================

model = SpeechCNN()

model.load_state_dict(
    torch.load(
        "results/speech_model.pth",
        map_location=torch.device("cpu")
    )
)

model.eval()

print("Speech Model Loaded Successfully")


# ==========================
# EMOTION MAP
# ==========================

emotion_map = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "surprise",
    6: "sad"
}


# ==========================
# INPUT AUDIO
# ==========================

audio_path = input("Enter audio path: ")

audio, sr = librosa.load(
    audio_path,
    sr=16000
)


# ==========================
# MFCC EXTRACTION
# ==========================

mfcc = librosa.feature.mfcc(
    y=audio,
    sr=sr,
    n_mfcc=40
)

if mfcc.shape[1] < 100:

    pad_width = 100 - mfcc.shape[1]

    mfcc = np.pad(
        mfcc,
        ((0, 0), (0, pad_width)),
        mode="constant"
    )

else:

    mfcc = mfcc[:, :100]


# ==========================
# TENSOR CONVERSION
# ==========================

x = torch.tensor(
    mfcc,
    dtype=torch.float32
).unsqueeze(0)

print("Input Shape:", x.shape)


# ==========================
# PREDICTION
# ==========================

with torch.no_grad():

    output = model(x)

    prediction = torch.argmax(
        output,
        dim=1
    ).item()

print(
    "\nPredicted Emotion:",
    emotion_map[prediction]
)