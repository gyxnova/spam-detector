
import joblib
import pandas as pd
from functools import lru_cache

from config import MODEL_PATH, VECTORIZER_PATH
from src.data.make_dataset import clean_text
from src.features.build_features import build_feature_matrix, add_hand_crafted_features, HAND_CRAFTED_COLS


@lru_cache(maxsize=1)
def _load_artifacts():
    """Loads the trained model and fitted vectorizer once, caches them in memory."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str) -> dict:
    """
    Takes a raw message string, returns a prediction dict.

    TODO :
    1. Load model + vectorizer via _load_artifacts()
    2. Clean the input text using clean_text()
    3. Build a single-row dataframe with the cleaned text
    4. Add hand-crafted features (add_hand_crafted_features)
    5. Build the feature matrix using build_feature_matrix(df, vectorizer=vectorizer, fit=False)
    6. Run model.predict() and model.predict_proba() on it
    7. Return a dict like:
       {
           "label": "spam" or "ham",
           "is_spam": True/False,
           "confidence": float (0-1)
       }
    """
    raise NotImplementedError("predict() logic to be implemented")
