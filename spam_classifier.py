"""
📧 Spam Email / SMS Classifier
================================
Compares MultinomialNaiveBayes vs Logistic Regression on TF-IDF features.
Dataset: SMS Spam Collection (UCI / 5572 messages)
Author: Anmol Pandey (AnmolPandey9119)
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import urllib.request
import re
import string
import os

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)

# ─── 1. LOAD DATASET ────────────────────────────────────────────────
print("=" * 60)
print("📧  SPAM EMAIL / SMS CLASSIFIER")
print("=" * 60)

DATA_URL = ("https://raw.githubusercontent.com/justmarkham/"
            "pycon-2016-tutorial/master/data/sms.tsv")
LOCAL_PATH = "sms_spam.tsv"

if not os.path.exists(LOCAL_PATH):
    print("\n[INFO] Downloading SMS Spam Collection dataset...")
    urllib.request.urlretrieve(DATA_URL, LOCAL_PATH)
    print("[INFO] Download complete.")

df = pd.read_csv(LOCAL_PATH, sep="\t", header=None, names=["label", "text"])
print(f"\n[DATA] Total messages : {len(df)}")
print(f"[DATA] Ham  (legit)   : {(df['label']=='ham').sum()}")
print(f"[DATA] Spam           : {(df['label']=='spam').sum()}")


# ─── 2. PREPROCESSING ───────────────────────────────────────────────
def preprocess(text: str) -> str:
    """Lowercase, remove punctuation and digits, strip whitespace."""
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

df["clean_text"] = df["text"].apply(preprocess)
df["label_enc"]  = (df["label"] == "spam").astype(int)   # 1 = spam, 0 = ham


# ─── 3. TRAIN / TEST SPLIT ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label_enc"],
    test_size=0.20, random_state=42, stratify=df["label_enc"]
)

print(f"\n[SPLIT] Train : {len(X_train)}  |  Test : {len(X_test)}")


# ─── 4. TF-IDF VECTORIZATION ────────────────────────────────────────
tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),   # unigrams + bigrams
    stop_words="english"
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)


# ─── 5. MODEL TRAINING ──────────────────────────────────────────────
models = {
    "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
    "Logistic Regression    ": LogisticRegression(max_iter=1000, C=5, random_state=42),
}

best_model, best_f1, best_name = None, 0, ""

print("\n" + "─" * 60)
print(f"{'Model':<30} {'Accuracy':>9} {'Precision':>10} {'Recall':>8} {'F1':>8}")
print("─" * 60)

for name, model in models.items():
    model.fit(X_train_tfidf, y_train)
    y_pred = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)

    print(f"{name:<30} {acc:>9.4f} {prec:>10.4f} {rec:>8.4f} {f1:>8.4f}")

    if f1 > best_f1:
        best_f1, best_model, best_name = f1, model, name.strip()

print("─" * 60)
print(f"\n🏆  Best Model: {best_name}  (F1 = {best_f1:.4f})\n")


# ─── 6. DETAILED EVALUATION (best model) ────────────────────────────
y_pred_best = best_model.predict(X_test_tfidf)

print("─" * 60)
print(f"DETAILED REPORT — {best_name}")
print("─" * 60)
print(classification_report(y_test, y_pred_best, target_names=["Ham", "Spam"]))

cm = confusion_matrix(y_test, y_pred_best)
tn, fp, fn, tp = cm.ravel()
print(f"Confusion Matrix:")
print(f"  True Negatives  (Ham   → Ham)  : {tn}")
print(f"  False Positives (Ham   → Spam) : {fp}  ← legit emails wrongly flagged")
print(f"  False Negatives (Spam  → Ham)  : {fn}  ← spam that slipped through")
print(f"  True Positives  (Spam  → Spam) : {tp}")


# ─── 7. CROSS-VALIDATION ────────────────────────────────────────────
cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring="f1")
print(f"\n5-Fold CV F1 Scores : {[round(s,4) for s in cv_scores]}")
print(f"Mean CV F1          : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ─── 8. LIVE PREDICTION DEMO ────────────────────────────────────────
print("\n" + "─" * 60)
print("LIVE PREDICTION DEMO")
print("─" * 60)

sample_messages = [
    "Congratulations! You've won a FREE iPhone. Click now to claim your prize!",
    "Hey, are we still on for lunch tomorrow at 1pm?",
    "URGENT: Your bank account has been suspended. Call this number immediately.",
    "Can you send me the notes from today's lecture? Thanks!",
    "WIN cash prizes worth £1000! Text WIN to 80080. No purchase necessary.",
]

for msg in sample_messages:
    clean  = preprocess(msg)
    vec    = tfidf.transform([clean])
    pred   = best_model.predict(vec)[0]
    label  = "🚨 SPAM" if pred == 1 else "✅ HAM"
    print(f"  {label}  |  {msg[:65]}{'...' if len(msg)>65 else ''}")

print("\n" + "=" * 60)
print("✅  Classifier training complete!")
print("=" * 60)
