# Spam Detector

An end-to-end spam detection system for **SMS and Email**, combining a classical
ML classifier with an LLM-powered explanation layer. Built collaboratively,
deployed on a zero-cost stack.

**🔗 Live demo:** https://gyxnova.github.io/spam-detector/
**🔗 API:** https://spam-detector-5k43.onrender.com/docs

## Architecture
Raw data (SMS + Email, ~88k messages)
↓
Preprocessing & cleaning (HTML/URL/email stripping)
↓
TF-IDF vectorization (unigrams + bigrams, 15k features)
↓
Logistic Regression classifier (97% F1)
↓
FastAPI backend (/predict, /explain, /health)
↓
Groq LLM (openai/gpt-oss-20b) — natural language explanations
↓
Static frontend (GitHub Pages) ← → Backend (Render)

## Stack

- **ML**: scikit-learn (TF-IDF + Logistic Regression), pandas
- **API**: FastAPI, deployed on Render (free tier)
- **LLM layer**: Groq API (openai/gpt-oss-20b) for explanation generation
- **Frontend**: Vanilla HTML/CSS/JS, deployed on GitHub Pages
- **No DVC** — model artifacts are small enough (~700KB total) to commit directly to git

## Project structure
├── data/
│ ├── raw/ # sms_spam.csv, email_spam.csv (tracked via Git LFS)
│ └── processed/ # spam_unified1.csv — unified, cleaned dataset
├── src/
│ ├── data/make_dataset.py # loads, cleans, unifies SMS + email data
│ ├── features/build_features.py # TF-IDF feature pipeline
│ └── models/
│ ├── train_model.py # trains & evaluates classifier
│ ├── predict_model.py # inference helper
│ └── explain.py # Groq LLM explanation layer
├── api/main.py # FastAPI service
├── frontend/index.html # UI (source of truth)
├── docs/index.html # copy served by GitHub Pages
├── config.py # central paths & hyperparameters
├── models/ # trained model, vectorizer, metadata
└── requirements.txt

## Results

- **97% F1-score** on held-out test set (17,723 messages)
- Logistic Regression outperformed Multinomial Naive Bayes (0.969 vs 0.947 CV F1)
- Trained on 88,615 unified messages (SMS: UCI SMS Spam Collection, Email: TREC2007 + Enron-Spam)

## A debugging story worth mentioning

An earlier version of this pipeline added hand-crafted features (URL count,
exclamation marks, capital-letter ratio, digit ratio, message length) alongside
TF-IDF, hoping to boost accuracy. Instead, it caused **obvious spam to be
confidently misclassified as ham** — e.g. "WIN A FREE IPHONE NOW CLICK HERE"
scored 99.7% ham.

Root cause, found by inspecting the model's \decision_function\ output directly:
the \capital_ratio\ feature had learned an **inverted relationship** with spam.
Since ~94% of the training data was email (rarely 100% capitalized) and only
~6% was SMS, the rare all-caps messages in training happened to be ham more
often than spam — the model learned a real but non-generalizable pattern from
this specific dataset's skew.

Standard fixes (StandardScaler, then clipping outliers to ±3 std devs) reduced
but didn't eliminate the instability. Ultimately, dropping the hand-crafted
features entirely and using **TF-IDF alone** fixed the issue completely, with
no meaningful F1 loss (0.9718 → 0.9687) — confirming the extra features were
adding risk without real signal.

## Setup

```bash
git clone https://github.com/gyxnova/spam-detector.git
cd spam-detector
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e .
pip install -r requirements.txt
cp .env.example .env       # add your GROQ_API_KEY
```

## Pipeline

```bash
python -m src.data.make_dataset       # build unified dataset
python -m src.features.build_features # build TF-IDF features
python -m src.models.train_model      # train & save model
uvicorn api.main:app --reload         # run API locally
```

Open \rontend/index.html\ in a browser (update \API_BASE\ to \http://127.0.0.1:8000\
for local testing).

## API

| Endpoint    | Method | Description                                  |
|-------------|--------|-----------------------------------------------|
| \/\         | GET    | API status message                            |
| \/health\   | GET    | Liveness check                                |
| \/predict\  | POST   | \{text}\ → label, confidence                  |
| \/explain\  | POST   | \{text}\ → label, confidence, LLM explanation |

## Deployment

- **API**: Render (free tier, auto-deploys from \main\)
- **Frontend**: GitHub Pages, serving \docs/index.html\
- **LLM**: Groq API (free tier)

## Known limitations

- Genuinely ambiguous promotional messages (e.g. legitimate marketing texts
  with discount codes) are sometimes misclassified — spam and real marketing
  copy share a lot of surface-level vocabulary, and this is a known hard
  problem in spam detection generally, not specific to this model.
- Free-tier Render hosting spins down after 15 minutes of inactivity; the
  first request after idling takes 30-60 seconds to wake up.
- Training data skews heavily toward email (~94%) vs SMS (~6%), which may
  affect relative performance across the two message types.
