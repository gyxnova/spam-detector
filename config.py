"""
Central configuration for the spam detector project.
All scripts should import paths and hyperparameters from here
instead of hardcoding them, so there's one source of truth.
"""
from pathlib import Path

# --- Project root & data paths ---
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

SMS_RAW_PATH = DATA_RAW_DIR / "sms_spam.csv"
EMAIL_RAW_PATH = DATA_RAW_DIR / "email_spam.csv"
PROCESSED_DATA_PATH = DATA_PROCESSED_DIR / "spam_unified1.csv"

# --- Model artifact paths ---
MODELS_DIR = PROJECT_ROOT / "models"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
MODEL_PATH = MODELS_DIR / "spam_clf.joblib"
METADATA_PATH = MODELS_DIR / "metadata.joblib"

# --- Feature engineering ---
HAND_CRAFTED_COLS = ["url_count", "exclamation_count", "capital_ratio", "digit_ratio", "message_length"]

TFIDF_PARAMS = {
    "max_features": 15000,
    "ngram_range": (1, 2),
    "stop_words": "english",
    "min_df": 2,
}

# --- Training ---
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
