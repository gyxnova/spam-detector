import re
import pandas as pd
import joblib
from pathlib import Path
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

MODELS_DIR = Path("models")
HAND_CRAFTED_COLS = ["url_count", "exclamation_count", "capital_ratio", "digit_ratio", "message_length"]


def url_count(text):
    """Number of URL tokens in the (already-cleaned) message."""
    return len(re.findall(r"http\S+|www\.\S+", text))


def exclamation_count(text):
    """Number of ! characters."""
    return text.count("!")


def capital_ratio(text):
    """Ratio of uppercase letters to all alphabetic letters."""
    letters = [c for c in text if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    capitals = [c for c in letters if c.isupper()]
    return len(capitals) / len(letters)


def digit_ratio(text):
    """Ratio of digits to total characters."""
    if len(text) == 0:
        return 0.0
    digits = [c for c in text if c.isdigit()]
    return len(digits) / len(text)


def message_length(text):
    """Total number of characters."""
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
        vectorizer = TfidfVectorizer(
            max_features=15000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2,
        )
        tfidf = vectorizer.fit_transform(df["text"])
    else:
        tfidf = vectorizer.transform(df["text"])

    hand_crafted = df[HAND_CRAFTED_COLS].values
    extra = csr_matrix(hand_crafted)
    features = hstack([tfidf, extra]).tocsr()
    return features, vectorizer


def save_vectorizer(vectorizer: TfidfVectorizer, path: Path = MODELS_DIR / "vectorizer.joblib"):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path: Path = MODELS_DIR / "vectorizer.joblib") -> TfidfVectorizer:
    return joblib.load(path)


if __name__ == "__main__":
    df = pd.read_csv("data/processed/spam_dataset.csv")

    if not all(col in df.columns for col in HAND_CRAFTED_COLS):
        df = add_hand_crafted_features(df)

    X, vec = build_feature_matrix(df, fit=True)
    save_vectorizer(vec)
    print(f"Final feature matrix shape: {X.shape}")