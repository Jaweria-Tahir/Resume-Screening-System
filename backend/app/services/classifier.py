
from pathlib import Path
import joblib

from app.services.preprocess import clean_text

MODEL_DIR = Path(__file__).resolve().parent.parent / "ml_models"

_model = joblib.load(MODEL_DIR / "classifier.pkl")
_vectorizer = joblib.load(MODEL_DIR / "vectorizer.pkl")
_label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")

def predict_category(resume_text: str, top_n: int = 3) -> dict:
  
    cleaned = clean_text(resume_text)
    vector = _vectorizer.transform([cleaned])

    probabilities = _model.predict_proba(vector)[0]
    top_indices = probabilities.argsort()[::-1][:top_n]

    top_predictions = [
        {
            "category": _label_encoder.inverse_transform([idx])[0],
            "probability": round(float(probabilities[idx]), 4),
        }
        for idx in top_indices
    ]

    CONFIDENCE_THRESHOLD = 0.35
    is_confident = top_predictions[0]["probability"] >= CONFIDENCE_THRESHOLD
    return {
    "predicted_category": top_predictions[0]["category"] if is_confident else "Unknown / Not a resume",
    "confidence": top_predictions[0]["probability"],
    "top_predictions": top_predictions,
    "is_confident": is_confident,
}
