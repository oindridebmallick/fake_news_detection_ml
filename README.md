# fake_news_detection_ml
This document outlines the approach for building the Fake News Detection ML model using Natural Language Processing (NLP) and traditional Machine Learning algorithms.

User Review Required
NOTE

Dataset: For a highly accurate model, we need a dataset. A common starting point is to use the ISOT Fake News Dataset (or a similar dataset from Kaggle) which consists of two files: Fake.csv and True.csv. 

Proposed Architecture
Environment & Dependencies: We'll use pandas for data manipulation, scikit-learn for ML models and TF-IDF, and nltk or standard regex for NLP text cleaning.
Preprocessing Pipeline:
Text Lowercasing
Removing punctuation, special characters, and URLs
Stopword removal
Feature Extraction:
Convert cleaned text to numerical data using TfidfVectorizer (Term Frequency-Inverse Document Frequency).
Model Selection:
Logistic Regression: Our primary model as it's highly interpretable, fast, and often performs excellently on binary text classification.
Naive Bayes (MultinomialNB): A great secondary baseline for NLP text classification.
Serialization: Save the trained model and the TF-IDF vectorizer using joblib so we don't have to retrain it every time we want to check a news article.

Data Layer
create a directory data/ to store the raw datasets.
Codebase
[NEW] requirements.txt
Dependencies: pandas, scikit-learn, nltk, joblib
[NEW] src/preprocess.py
Functions to clean and preprocess raw news text.
[NEW] src/train.py
Script to load data, run preprocessing, fit the TF-IDF vectorizer, train the Logistic Regression model, evaluate its accuracy, and save the artifacts.
[NEW] src/predict.py
Script that loads the trained model and vectorizer to predict whether a given input string is "Fake" or "Real".
Verification Plan
Automated Tests
The train.py script will automatically output performance metrics including Accuracy, Precision, Recall, F1-Score, and a Confusion Matrix on a reserved testing test (e.g., 20% of the dataset).
Manual Verification
We will use predict.py to manually pass in sample news headlines/articles (both famously fake and demonstrably true) and observe the model's prediction and confidence score.
