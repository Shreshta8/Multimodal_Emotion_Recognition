import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


emotion_texts = {
    "angry": [
        "I am angry",
        "I am furious",
        "I am upset",
        "I am irritated",
        "I am mad",
        "This makes me angry",
        "I feel rage",
        "I am frustrated",
        "I am annoyed",
        "I am very upset"
    ],

    "happy": [
        "I am happy",
        "I feel wonderful",
        "I am joyful",
        "I am delighted",
        "I am excited",
        "Today is amazing",
        "I feel great",
        "I am cheerful",
        "I am pleased",
        "Life is beautiful"
    ],

    "sad": [
        "I am sad",
        "I feel depressed",
        "I am unhappy",
        "I feel lonely",
        "I feel miserable",
        "I am heartbroken",
        "Today is awful",
        "I feel down",
        "I am disappointed",
        "I feel terrible",
        "Nothing is going right",
        "I failed my exam",
        "I feel hopeless",
        "I want to cry",
        "I feel broken"
    ],

    "fear": [
        "I am scared",
        "I am afraid",
        "I feel nervous",
        "I feel anxious",
        "I am worried",
        "I feel terrified",
        "I am frightened",
        "I feel unsafe",
        "I am concerned",
        "I feel panic"
    ],

    "disgust": [
        "This is disgusting",
        "I feel disgusted",
        "That is gross",
        "This makes me sick",
        "I feel revolted",
        "This is unpleasant",
        "That is nasty",
        "I feel uncomfortable",
        "This is horrible",
        "I hate this"
        "I feel like vomiting",
        "This smells awful",
        "This food is rotten",
        "I want to throw up",
        "This is disgusting"
    ],

    "surprise": [
        "I am surprised",
        "That was unexpected",
        "I cannot believe it",
        "What a surprise",
        "This is shocking",
        "I am amazed",
        "That caught me off guard",
        "I am astonished",
        "This is unbelievable",
        "I did not expect that"
    ],

    "neutral": [
        "I am okay",
        "I feel normal",
        "Nothing special happened",
        "Today is ordinary",
        "I am fine",
        "Everything is normal",
        "It is a regular day",
        "I have no strong feelings",
        "Things are okay",
        "I am reading a book",
        "I am going to the market",
        "I am sitting in class",
        "I am working on my project",
        "I am using my computer",
        "Today is Monday",
        "The weather is normal",
        "I feel neutral"
    ]
}

training_data = [
    ("I am angry", "angry"),
    ("I am furious", "angry"),
    ("This makes me mad", "angry"),
    ("I am upset", "angry"),

    ("I am sad", "sad"),
    ("I am depressed", "sad"),
    ("Nothing is going right", "sad"),
    ("I feel lonely", "sad"),

    ("I am scared", "fear"),
    ("I am terrified", "fear"),
    ("I am anxious", "fear"),

    ("I feel disgusted", "disgust"),
    ("This makes me sick", "disgust"),
    ("I feel like vomiting", "disgust"),

    ("Today is Monday", "neutral"),
    ("I am reading a book", "neutral"),
    ("I am going to the market", "neutral"),

    ("I am happy", "happy"),
    ("Life is beautiful", "happy"),
    ("I feel wonderful", "happy"),

    ("Wow!", "surprise"),
    ("I did not expect that", "surprise"),
    ("That is surprising", "surprise")
]

data = []

for emotion, texts in emotion_texts.items():
    for text in texts:
        for _ in range(40):
            data.append({
                "text": text,
                "emotion": emotion
            })

for text, emotion in training_data:
    data.append({
        "text": text,
        "emotion": emotion
    })


df = pd.DataFrame(data)

x = df["text"]
y = df["emotion"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()

x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)

print("Training text emotion model...")
model.fit(x_train_vector, y_train)

predictions = model.predict(x_test_vector)

accuracy = accuracy_score(y_test, predictions)

print("\nText Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

with open("text_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("text_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nText model saved successfully.")