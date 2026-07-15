
import re

import string

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

DATA_PATH = "ml_training/data/UpdatedResumeDataSet.csv"
MODEL_DIR = "app/ml_models"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)          # URLs
    text = re.sub(r"\S+@\S+", " ", text)                  # emails
    text = re.sub(r"[^a-z\s]", " ", text)                 # punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()               # extra whitespace
    return text


def main():
    print("1) Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["Resume", "Category"])

    print(f"   {len(df)} resumes, {df['Category'].nunique()} categories")

    print("2) Cleaning text...")
    df["cleaned"] = df["Resume"].apply(clean_text)

    print("3) Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])

    print("4) Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned"], y, test_size=0.2, random_state=42, stratify=y
    )

    print("5) Vectorizing text (TF-IDF)...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("6) Training classifier (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_vec, y_train)

    print("7) Evaluating...")
    preds = model.predict(X_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"   Test accuracy: {acc:.3f}")
    print(classification_report(
        y_test, preds, target_names=label_encoder.classes_, zero_division=0
    ))

    print("8) Saving artifacts...")
    joblib.dump(model, f"{MODEL_DIR}/classifier.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
    print(f"   Saved to {MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
