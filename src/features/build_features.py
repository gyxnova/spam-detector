import re
import pandas as pd
import joblib
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from config import PROCESSED_DATA_PATH, VECTORIZER_PATH, HAND_CRAFTED_COLS, TFIDF_PARAMS


def url_count(text):
    """Number of URL tokens in the (already-cleaned) message."""
    return len(re.findall(r"http\S+|www\.\S+", text))


def exclamation_count(text):
    return text.count("!")


def capital_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    capitals = [c for c in letters if c.isupper()]
    return len(capitals) / len(letters)


def digit_ratio(text):
    if len(text) == 0:
        return 0.0
    digits = [c for c in text if c.isdigit()]
    return len(digits) / len(text)


def message_length(text):
    return len(text)


def add_hand_crafted_features(df: pd.DataFrame) -> pd.DataFrame:
    df["url_count"] = df["text"].apply(url_count)
    df["exclamation_count"] = df["text"].apply(exclamation_count)
    df["capital_ratio"] = df["text"].apply(capital_ratio)
    df["digit_ratio"] = df["text"].apply(digit_ratio)
    df["message_length"] = df["text"].apply(message_length)
    return df


def build_feature_matrix(df: pd.DataFrame, vectorizer: TfidfVectorizer = None, fit: bool = True):
    """Returns (feature_matrix, fitted_vectorizer)."""
    if fit:
        vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
        tfidf = vectorizer.fit_transform(df["text"])
    else:
        tfidf = vectorizer.transform(df["text"])

    hand_crafted = df[HAND_CRAFTED_COLS].values
    extra = csr_matrix(hand_crafted)
    features = hstack([tfidf, extra]).tocsr()
    return features, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path=VECTORIZER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path=VECTORIZER_PATH) -> TfidfVectorizer:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DATA_PATH)

    if not all(col in df.columns for col in HAND_CRAFTED_COLS):
        df = add_hand_crafted_features(df)

    X, vec = build_feature_matrix(df, fit=True)
    save_vectorizer(vec)
    print(f"Final feature matrix shape: {X.shape}")