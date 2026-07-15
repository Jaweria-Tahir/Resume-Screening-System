
import re
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn.calibration import CalibratedClassifierCV

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
    raw_rows = len(df)

    # --- FIX: the raw CSV is ~83% exact-duplicate rows (796 of 962). Training
    # on duplicates and then doing a random train/test split leaks the exact
    # same resume text into both train and test, which is why the old script
    # reported ~99.5% accuracy. That number was not real. Dedupe first so
    # every evaluation below reflects genuinely unseen text. ---
    df = df.drop_duplicates(subset=["Resume"]).reset_index(drop=True)
    print(f"   {raw_rows} raw rows -> {len(df)} unique resumes after de-duplication, "
          f"{df['Category'].nunique()} categories")

    counts = df["Category"].value_counts()
    print("   Unique examples per category (smallest 5):")
    print("   " + counts.tail(5).to_string().replace("\n", "\n   "))

    print("\n2) Cleaning text...")
    df["cleaned"] = df["Resume"].apply(clean_text)

    print("3) Encoding labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["Category"])
    X = df["cleaned"]

    # With some categories down to 3-7 unique examples, a single 80/20 split
    # is too noisy to trust (test set might have 0-1 examples of a class).
    # Use k-fold cross-validation instead so every example gets evaluated
    # while unseen, and honest accuracy is averaged over multiple folds.
    n_splits = min(3, counts.min())  # smallest class has 3 examples -> 3 folds
    print(f"\n4) Evaluating with {n_splits}-fold stratified cross-validation "
          f"(honest, leakage-free numbers)...")

    vectorizer_cv = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=1
    )
    X_vec_cv = vectorizer_cv.fit_transform(X)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for name, clf in [
        ("Logistic Regression", LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")),
        ("Linear SVM", LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)),
    ]:
        preds = cross_val_predict(clf, X_vec_cv, y, cv=skf)
        acc = accuracy_score(y, preds)
        f1 = f1_score(y, preds, average="macro")
        print(f"   {name}: accuracy={acc:.3f}  macro-F1={f1:.3f}")

    print("\n   (These are the numbers to actually trust. They will look a lot")
    print("   lower than the old 0.995 -- that's expected and correct: this")
    print("   is what happens on genuinely unseen resumes with ~166 unique")
    print("   training examples across 25 classes. See README for how to fix it.)")

    # --- Final model: train on ALL unique data (no held-out split, since we
    # already got an honest performance estimate above via cross-validation).
    # LinearSVC generally handles small, high-dimensional TF-IDF data better
    # than plain LogisticRegression, so we use it here, wrapped so it can
    # still output probabilities for the API's "confidence" field. ---
    print("\n5) Fitting final vectorizer + model on all available data...")
    vectorizer = TfidfVectorizer(
        max_features=5000, stop_words="english", ngram_range=(1, 2), min_df=1
    )
    X_vec = vectorizer.fit_transform(X)

    base_svm = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000)
    model = CalibratedClassifierCV(base_svm, cv=n_splits)
    model.fit(X_vec, y)

    print("6) Saving artifacts...")
    joblib.dump(model, f"{MODEL_DIR}/classifier.pkl")
    joblib.dump(vectorizer, f"{MODEL_DIR}/vectorizer.pkl")
    joblib.dump(label_encoder, f"{MODEL_DIR}/label_encoder.pkl")
    print(f"   Saved to {MODEL_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
