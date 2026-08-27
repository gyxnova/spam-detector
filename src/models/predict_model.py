"""
Inference helper used by the FastAPI service.
Loads the trained classifier, vectorizer, and scaler once and exposes a predict() function.
"""
import joblib
import pandas as pd
from functools import lru_cache

from config import MODEL_PATH, VECTORIZER_PATH, MODELS_DIR
from src.data.make_dataset import clean_text
from src.features.build_features import build_feature_matrix, add_hand_crafted_features

SCALER_PATH = MODELS_DIR / "scaler.joblib"


@lru_cache(maxsize=1)
def _load_artifacts():
    """Loads the trained model, fitted vectorizer, and fitted scaler once, caches them in memory."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, vectorizer, scaler


def predict(text: str) -> dict:
    """Classify one raw message as spam or ham.

    The returned confidence is the model probability of the predicted class.
    """
    model, vectorizer, scaler = _load_artifacts()

    df = pd.DataFrame({"text": [clean_text(text)]})
    df = add_hand_crafted_features(df)
    features, _, _ = build_feature_matrix(df, vectorizer=vectorizer, scaler=scaler, fit=False)

    prediction = int(model.predict(features)[0])
    probability_index = list(model.classes_).index(prediction)
    confidence = float(model.predict_proba(features)[0][probability_index])
    is_spam = prediction == 1

    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "confidence": confidence,
    }