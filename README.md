# Multimodal_Emotion_Recognition

Speech and Text Based Multimodal Emotion Recognition System

## Project Overview

This project is a Multimodal Emotion Recognition System that identifies human emotions using speech, text, and a fusion of both modalities. The system combines speech-based and text-based emotion analysis to improve emotion recognition performance.

## Emotions Detected

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

## System Architecture

### Speech Emotion Recognition
- Feature Extraction: MFCC (Mel-Frequency Cepstral Coefficients)
- Model: CNN (Convolutional Neural Network)

### Text Emotion Recognition
- Feature Extraction: TF-IDF
- Model: Logistic Regression

### Fusion Emotion Recognition
- Feature-Level Fusion of speech and text features
- Model: Random Forest Classifier

## How to Run

1. Install the required libraries:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

## Technologies Used

- Python
- PyTorch
- Scikit-learn
- Librosa
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Results

The project includes:

- Confusion Matrix
- Emotion Distribution Analysis
- Model Accuracy Comparison
- Speech Emotion Clusters
- Text Emotion Clusters
- Fusion Emotion Clusters

## Project Structure

- models/ : Speech, Text, and Fusion pipelines
- results/ : Visualizations and evaluation results
- app.py : Main application file
- requirements.txt : Required dependencies
- README.md : Project documentation
- Model files (.pth/.pkl) : Trained models
- Report PDF : Project report

## Author

Shreshta

B.Tech Project – Multimodal Emotion Recognition
