import pandas as pd
from src.data.make_dataset import clean_text
from src.features.build_features import build_feature_matrix, add_hand_crafted_features, load_vectorizer, load_scaler
from config import MODEL_PATH
import joblib

model = joblib.load(MODEL_PATH)
vectorizer = load_vectorizer()
scaler = load_scaler()

text = 'WIN A FREE IPHONE NOW CLICK HERE'
cleaned = clean_text(text)
df = pd.DataFrame({'text': [cleaned]})
df = add_hand_crafted_features(df)

print('Raw hand-crafted values:')
print(df[['url_count', 'exclamation_count', 'capital_ratio', 'digit_ratio', 'message_length']])

features, _, _ = build_feature_matrix(df, vectorizer=vectorizer, scaler=scaler, fit=False)
score = model.decision_function(features)[0]
print()
print('Raw decision score:', score)

coefs = model.coef_[0]
hand_crafted_coefs = coefs[-5:]
scaled_values = scaler.transform(df[['url_count', 'exclamation_count', 'capital_ratio', 'digit_ratio', 'message_length']].values)[0]
print('Scaled hand-crafted values:', scaled_values)
print('Coefficients:', hand_crafted_coefs)
contribution = sum(c * v for c, v in zip(hand_crafted_coefs, scaled_values))
print('Hand-crafted contribution to score:', contribution)

tfidf_only_score = score - contribution
print('TF-IDF-only contribution to score:', tfidf_only_score)
