# train_binary_xgb.py
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from joblib import dump

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (classification_report, roc_auc_score, f1_score,
                             precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay)
from imblearn.over_sampling import RandomOverSampler
from xgboost import XGBClassifier

RANDOM_STATE = 42
OUT_DIR = Path("reports/binary_xgb")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============
# 1. LOAD DATA
# ============
train_path = "data/processed/train.csv"
test_path  = "data/processed/test.csv"

train_df = pd.read_csv(train_path)
test_df  = pd.read_csv(test_path)

y_train = train_df["Rating"].map(lambda x: 0 if x in [1, 2] else 1).values
y_test  = test_df["Rating"].map(lambda x: 0 if x in [1, 2] else 1).values
X_train = train_df["CleanText"].astype(str)
X_test  = test_df["CleanText"].astype(str)

# ============
# 2. TF-IDF VECTORIZATION
# ============
tfidf = TfidfVectorizer(
    max_features=20000,
    sublinear_tf=True,
    ngram_range=(1,2),
    min_df=2,
    max_df=0.9
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# ============
# 3. OVERSAMPLING
# ============
ros = RandomOverSampler(random_state=RANDOM_STATE)
X_train_res, y_train_res = ros.fit_resample(X_train_tfidf, y_train)

# ============
# 4. TRAIN XGBOOST
# ============
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    eval_metric="logloss",
    n_jobs=-1
)
xgb.fit(X_train_res, y_train_res)

# ============
# 5. EVALUATION
# ============
y_proba = xgb.predict_proba(X_test_tfidf)[:, 1]
y_pred_05 = (y_proba >= 0.5).astype(int)

# Seuil optimal via courbe PR
prec, rec, thr = precision_recall_curve(y_test, y_proba)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
thr_opt = float(thr[np.nanargmax(f1s)])
y_pred_opt = (y_proba >= thr_opt).astype(int)

metrics = {
    "f1_0.5": float(f1_score(y_test, y_pred_05)),
    "f1_opt": float(f1_score(y_test, y_pred_opt)),
    "roc_auc": float(roc_auc_score(y_test, y_proba)),
    "threshold_opt": thr_opt
}

print("=== Classification Report (seuil 0.5) ===")
print(classification_report(y_test, y_pred_05, digits=3))
print("F1@0.5:", metrics["f1_0.5"])
print("F1@opt:", metrics["f1_opt"])
print("ROC-AUC:", metrics["roc_auc"])
print("Seuil optimal:", thr_opt)

# Confusion matrices
for tag, yhat in [("0.5", y_pred_05), ("opt", y_pred_opt)]:
    cm = confusion_matrix(y_test, yhat)
    disp = ConfusionMatrixDisplay(cm, display_labels=[0,1])
    disp.plot()
    plt.title(f"Confusion Matrix ({tag})")
    plt.savefig(OUT_DIR / f"confusion_matrix_{tag}.png")
    plt.close()

# ============
# 6. SHAP EXPLANATIONS
# ============
explainer = shap.Explainer(xgb)
shap_values = explainer(X_test_tfidf[:1000])  # limite à 1000 exemples pour le temps

# Résumé global
shap.summary_plot(shap_values, features=X_test_tfidf[:1000].toarray(),
                  feature_names=tfidf.get_feature_names_out(),
                  show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_summary.png")
plt.close()

# Exemple local
shap.plots.waterfall(shap_values[0], show=False)
plt.tight_layout()
plt.savefig(OUT_DIR / "shap_waterfall_example.png")
plt.close()

# ============
# 7. SAVE MODEL & METRICS
# ============
dump(xgb, "models/binary_xgb_model.joblib")
pd.Series(metrics).to_json(OUT_DIR / "metrics_xgb.json", indent=2)
print("✅ Model & reports saved in:", OUT_DIR.as_posix())
