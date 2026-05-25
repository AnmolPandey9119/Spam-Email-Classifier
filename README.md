# 📧 Spam Email / SMS Classifier

> **98.21% accuracy** on 5,572 real SMS messages — NLP pipeline with TF-IDF features and comparative model evaluation.

---

## 🎯 Project Overview

A production-ready NLP pipeline that classifies SMS/email messages as **spam or legitimate (ham)** using the classic **SMS Spam Collection dataset** from UCI. Compares two model architectures and automatically selects the best performer.

| Metric | Multinomial NB | Logistic Regression |
|--------|:--------------:|:-------------------:|
| Accuracy | **98.21%** | 98.21% |
| Precision (Spam) | 97.78% | **100%** |
| Recall (Spam) | **88.59%** | 86.58% |
| F1 Score (Spam) | **92.96%** | 92.81% |
| 5-Fold CV F1 | **93.10% ± 0.39%** | — |

---

## 📂 Dataset

**SMS Spam Collection** — 5,572 tagged messages (4,825 ham + 747 spam)  
Auto-downloaded on first run from the public repository.

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

**Pipeline:** Text cleaning → TF-IDF (5000 features, 1-2 ngrams) → Model training → Cross-validation

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/AnmolPandey9119/Spam-Email-Classifier.git
cd Spam-Email-Classifier

# Install dependencies
pip install -r requirements.txt

# Run the classifier
python spam_classifier.py
```

---

## 📊 Sample Output

```
📧  SPAM EMAIL / SMS CLASSIFIER
============================================================
[DATA] Total messages : 5572  |  Ham: 4825  |  Spam: 747

Model                    Accuracy  Precision  Recall     F1
Multinomial Naive Bayes  0.9821    0.9778     0.8859  0.9296  ← Best
Logistic Regression      0.9821    1.0000     0.8658  0.9281

LIVE PREDICTION DEMO
🚨 SPAM  |  Congratulations! You've won a FREE iPhone. Click now...
✅ HAM   |  Hey, are we still on for lunch tomorrow at 1pm?
🚨 SPAM  |  URGENT: Your bank account has been suspended...
✅ HAM   |  Can you send me the notes from today's lecture?
```

---

## 🔑 Key ML Concepts Demonstrated

- **TF-IDF Vectorization** with bigram features and stop-word removal
- **Multinomial Naive Bayes** — optimal for text classification with word-frequency features
- **Logistic Regression** — strong baseline with L2 regularization
- **Stratified Train/Test Split** — preserves class imbalance ratios
- **5-Fold Cross-Validation** — robust generalization estimate
- **Precision vs Recall tradeoff** — Logistic Regression has 100% precision (no false positives)

---

## 📈 Why This Matters

Spam detection is a **core NLP problem** in production systems. This project demonstrates:
- Handling **class imbalance** (87% ham vs 13% spam)
- Choosing the right metric (F1 > accuracy for imbalanced data)
- **Bigram features** to capture phrases like "free prize" and "click now"

---

## 👨‍💻 Author

**Anmol Pandey** — AI/ML Engineer  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=flat-square&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anmol-pandey-240105376)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/AnmolPandey9119)

---

*Part of my AI/ML project portfolio — targeting ML Engineer & Data Scientist roles.*
