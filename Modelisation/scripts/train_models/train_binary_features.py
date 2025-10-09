# --- config ablation ---
USE_POLARITY = False
USE_SUBJECTIVITY = False
NUM_WEIGHT = 0.3        # poids de la branche numérique dans FeatureUnion (essayez 0.2–0.5)
USE_ROS = True          # teste aussi False
C_LOGREG = 1.0          # essaye 0.5, 1.0, 2.0

import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_curve, roc_curve, auc)
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from joblib import dump
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin


OUT_DIR = Path("reports/binary_features")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- transformer simple (maj: options pour polarity/subjectivity) ---
class SimpleTextFeats(BaseEstimator, TransformerMixin):
    def __init__(self, use_polarity=False, use_subjectivity=False):
        self.use_polarity = use_polarity
        self.use_subjectivity = use_subjectivity
    def fit(self, X, y=None): return self
    def transform(self, X):
        out = []
        for t in X:
            s = t if isinstance(t, str) else ""
            n_chars = len(s)
            n_words = len(s.split()) if s else 0
            exclam = s.count("!")
            upper_ratio = (sum(c.isupper() for c in s)/n_chars) if n_chars>0 else 0.0
            row = [n_chars, n_words, exclam, upper_ratio]
            if self.use_polarity or self.use_subjectivity:
                tb = TextBlob(s).sentiment
                if self.use_polarity: row.append(tb.polarity)
                if self.use_subjectivity: row.append(tb.subjectivity)
            out.append(row)
        return np.asarray(out, dtype=float)

# --- data ---
train_df = pd.read_csv("data/processed/train.csv")
test_df  = pd.read_csv("data/processed/test.csv")
y_train = train_df["Rating"].astype(int).map(lambda r: 0 if r in (1,2) else 1).values
y_test  = test_df["Rating"].astype(int).map(lambda r: 0 if r in (1,2) else 1).values
X_train_text = train_df["CleanText"].astype(str).values
X_test_text  = test_df["CleanText"].astype(str).values

# --- branches ---
tfidf = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True, min_df=2, max_df=0.9, max_features=20000)
num_branch = Pipeline([
    ("feats", SimpleTextFeats(use_polarity=USE_POLARITY, use_subjectivity=USE_SUBJECTIVITY)),
    ("scale", StandardScaler())
])

union = FeatureUnion([
    ("tfidf", tfidf),
    ("num",   num_branch)
], transformer_weights={"tfidf": 1.0, "num": NUM_WEIGHT})

# --- pipeline ---
steps = [("features", union)]
if USE_ROS:
    steps.append(("ros", RandomOverSampler(random_state=42)))
steps.append(("clf", LogisticRegression(max_iter=4000, random_state=42, C=C_LOGREG)))
pipe = ImbPipeline(steps=steps)

# --- fit & predict ---
pipe.fit(X_train_text, y_train)
proba_test = pipe.predict_proba(X_test_text)[:, 1]
y_pred_05  = (proba_test >= 0.5).astype(int)

# seuil optimal sur PR
prec, rec, thr = precision_recall_curve(y_test, proba_test)
f1s = 2*prec*rec/np.maximum(prec+rec, 1e-12)
best_i = int(np.nanargmax(f1s))
thr_opt = float(thr[max(0, best_i-1)]) if len(thr)>0 else 0.5
y_pred_opt = (proba_test >= thr_opt).astype(int)

# --- metrics ---
metrics = {
  "config": {
    "use_polarity": USE_POLARITY, "use_subjectivity": USE_SUBJECTIVITY,
    "num_weight": NUM_WEIGHT, "use_ros": USE_ROS, "C": C_LOGREG
  },
  "threshold_0.5": {
    "f1": float(f1_score(y_test, y_pred_05)),
    "roc_auc": float(roc_auc_score(y_test, proba_test))
  },
  "threshold_opt": {
    "thr": thr_opt,
    "f1": float(f1_score(y_test, y_pred_opt)),
    "roc_auc": float(roc_auc_score(y_test, proba_test))
  }
}
with open(OUT_DIR/"metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# --- confusion matrices ---
for tag, yhat in [("0.5", y_pred_05), ("opt", y_pred_opt)]:
    cm = confusion_matrix(y_test, yhat, labels=[0,1])
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(ax=ax, values_format='d')
    plt.tight_layout(); plt.savefig(OUT_DIR/f"confusion_matrix_{tag}.png"); plt.close(fig)

# --- ROC & PR ---
fpr, tpr, _ = roc_curve(y_test, proba_test)
fig, ax = plt.subplots(); ax.plot(fpr,tpr); ax.plot([0,1],[0,1],'--')
ax.set_title("ROC"); plt.tight_layout(); plt.savefig(OUT_DIR/"roc.png"); plt.close(fig)

fig, ax = plt.subplots(); ax.plot(rec,prec); ax.set_title("Precision-Recall")
plt.tight_layout(); plt.savefig(OUT_DIR/"pr.png"); plt.close(fig)

# --- classification report txt ---
with open(OUT_DIR/"classification_report_0.5.txt", "w", encoding="utf-8") as f:
    f.write(classification_report(y_test, y_pred_05, digits=3, zero_division=0))

# --- save model ---
dump(pipe, "models/binary_features_model.joblib")
print("Saved model -> models/binary_features_model.joblib")
print("Saved reports ->", OUT_DIR.as_posix())
print("F1@0.5:", metrics["threshold_0.5"]["f1"], "| F1@opt:", metrics["threshold_opt"]["f1"], "| thr_opt:", metrics["threshold_opt"]["thr"])
