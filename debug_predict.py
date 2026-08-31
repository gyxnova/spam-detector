import pandas as pd
from src.data.make_dataset import clean_text
from src.features.build_features import build_feature_matrix, add_hand_crafted_features, load_vectorizer
from config import MODEL_PATH
import joblib

model = joblib.load(MODEL_PATH)
vectorizer = load_vectorizer()

text = 'WIN A FREE IPHONE NOW CLICK HERE'
cleaned = clean_text(text)
df = pd.DataFrame({'text': [cleaned]})
df = add_hand_crafted_features(df)

print('Hand-crafted feature values:')
print(df[['url_count', 'exclamation_count', 'capital_ratio', 'digit_ratio', 'message_length']])

features, _ = build_feature_matrix(df, vectorizer=vectorizer, fit=False)
score = model.decision_function(features)[0]
print()
print('Raw decision score:', score, '(negative = ham, positive = spam)')

# isolate contribution of just the last 5 hand-crafted columns
coefs = model.coef_[0]
hand_crafted_coefs = coefs[-5:]
hand_crafted_values = df[['url_count', 'exclamation_count', 'capital_ratio', 'digit_ratio', 'message_length']].values[0]
contribution = sum(c * v for c, v in zip(hand_crafted_coefs, hand_crafted_values))
print('Hand-crafted features contribution to score:', contribution)
print('Coefficients for [url_count, exclamation_count, capital_ratio, digit_ratio, message_length]:')
print(hand_crafted_coefs)
