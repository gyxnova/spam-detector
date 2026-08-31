import pandas as pd
import joblib
from src.data.make_dataset import clean_text
from src.features.build_features import build_feature_matrix, add_hand_crafted_features, load_vectorizer, load_scaler
from config import MODEL_PATH

model = joblib.load(MODEL_PATH)
vectorizer = load_vectorizer()
scaler = load_scaler()

coefs = model.coef_[0]
hand_crafted_coefs = coefs[-5:]
print('Hand-crafted coefficients [url, exclaim, capital_ratio, digit_ratio, length]:')
print(hand_crafted_coefs)

tests = [
    'WIN A FREE IPHONE NOW CLICK HERE',
    'Free entry in 2 a wkly comp to win FA Cup final tkts',
    'WINNER!! You have been selected to receive a 900 pound prize',
]
for text in tests:
    cleaned = clean_text(text)
    df = pd.DataFrame({'text': [cleaned]})
    df = add_hand_crafted_features(df)
    features, _, _ = build_feature_matrix(df, vectorizer=vectorizer, scaler=scaler, fit=False)
    score = model.decision_function(features)[0]
    hc_scaled = scaler.transform(df[['url_count','exclamation_count','capital_ratio','digit_ratio','message_length']].values)[0]
    import numpy as np
    hc_clipped = np.clip(hc_scaled, -3, 3)
    contribution = sum(c*v for c,v in zip(hand_crafted_coefs, hc_clipped))
    print(f'{text[:40]:40} | score={score:.3f} | hc_contribution={contribution:.3f} | tfidf={score-contribution:.3f}')
