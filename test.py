import pickle

with open("text_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("text_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

while True:
    text = input("Enter text: ")

    text_features = vectorizer.transform([text])

    prediction = model.predict(text_features)

    print("Predicted Emotion:", prediction[0])