"""
Builds TF-IDF features for the spam classifier from the 'text' column.

Saves the fitted vectorizer to models/vectorizer.joblib for reuse at inference time.
"""
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PROCESSED_DATA_PATH, VECTORIZER_PATH, TFIDF_PARAMS


def build_feature_matrix(df: pd.DataFrame, vectorizer: TfidfVectorizer = None, fit: bool = True):
    """Returns (feature_matrix, fitted_vectorizer)."""
    if fit:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf = vectorizer.fit_transform(df["text"])
    else:
        tfidf = vectorizer.transform(df["text"])
    return tfidf, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path=VECTORIZER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path=VECTORIZER_PATH) -> TfidfVectorizer:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)
    X, vec = build_feature_matrix(df, fit=True)
    save_vectorizer(vec)
    print(f"Final feature matrix shape: {X.shape}")