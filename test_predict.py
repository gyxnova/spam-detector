from src.models.predict_model import predict

tests = [
    'WIN A FREE IPHONE NOW CLICK HERE',
    'Free entry in 2 a wkly comp to win FA Cup final tkts',
    'WINNER!! You have been selected to receive a 900 pound prize',
    'URGENT! You have won a 1 week FREE membership in our 100000 Prize Jackpot!',
    'Congratulations! You have won a free iPhone. Click here to claim your prize now!',
]

for t in tests:
    r = predict(t)
    print(r['label'], round(r['confidence'], 3), '|', t)
