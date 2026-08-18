
import joblib
import pandas as pd
from functools import lru_cache

from config import MODEL_PATH, VECTORIZER_PATH
from src.data.make_dataset import clean_text
from src.features.build_features import add_hand_crafted_features, build_feature_matrix


@lru_cache(maxsize=1)
def _load_artifacts():
    """Loads the trained model and fitted vectorizer once, caches them in memory."""
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def predict(text: str) -> dict:
    """Classify one raw message as spam or ham.

    The returned confidence is the model probability of the predicted class.
    """
    model, vectorizer = _load_artifacts()

    df = pd.DataFrame({"text": [clean_text(text)]})
    df = add_hand_crafted_features(df)
    features, _ = build_feature_matrix(df, vectorizer=vectorizer, fit=False)

    prediction = int(model.predict(features)[0])
    probability_index = list(model.classes_).index(prediction)
    confidence = float(model.predict_proba(features)[0][probability_index])
    is_spam = prediction == 1

    return {
        "label": "spam" if is_spam else "ham",
        "is_spam": is_spam,
        "confidence": confidence,
    }
