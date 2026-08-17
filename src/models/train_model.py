import logging

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from config import PROCESSED_DATA_PATH, MODEL_PATH, METADATA_PATH, TEST_SIZE, RANDOM_STATE, CV_FOLDS
from src.features.build_features import build_feature_matrix, save_vectorizer, HAND_CRAFTED_COLS

CANDIDATES = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "naive_bayes": MultinomialNB(),
}


def main():
    logger = logging.getLogger(__name__)
    df = pd.read_csv(PROCESSED_DATA_PATH)

    if not all(col in df.columns for col in HAND_CRAFTED_COLS):
        from src.features.build_features import add_hand_crafted_features
        df = add_hand_crafted_features(df)

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )

    X_train, vectorizer = build_feature_matrix(train_df, fit=True)
    X_test, _ = build_feature_matrix(test_df, vectorizer=vectorizer, fit=False)
    y_train, y_test = train_df["label"], test_df["label"]

    best_name, best_model, best_score = None, None, -1
    for name, model in CANDIDATES.items():
        scores = cross_val_score(model, X_train, y_train, cv=CV_FOLDS, scoring="f1")
        logger.info(f"{name}: CV F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")
        if scores.mean() > best_score:
            best_name, best_model, best_score = name, model, scores.mean()

    logger.info(f"Best model: {best_name} (CV F1={best_score:.4f}). Fitting on full train set.")
    best_model.fit(X_train, y_train)

    y_pred = best_model.predict(X_test)
    logger.info("\n" + classification_report(y_test, y_pred, target_names=["ham", "spam"]))
    logger.info(f"Confusion matrix:\n{confusion_matrix(y_test, y_pred)}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)
    save_vectorizer(vectorizer)
    joblib.dump({"model_name": best_name, "cv_f1": best_score}, METADATA_PATH)
    logger.info(f"Saved model + vectorizer to {MODEL_PATH.parent}/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    main()