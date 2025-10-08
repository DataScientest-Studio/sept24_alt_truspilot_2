# make_reports_from_predictions.py
# ------------------------------------------------------------
# Génère :
#  - Matrices de confusion multiclasse (brute + normalisée)
#  - Rapport de classification multiclasse (txt)
#  - Passage en binaire (1–2 = 0, 3–4–5 = 1)
#  - Matrices de confusion binaires (seuil 0.5 + seuil optimal F1)
#  - Courbes ROC et Precision–Recall
#  - Fichier JSON de métriques (F1, ROC-AUC, PR-AUC, seuil optimal)
#
# Exécution :
#   python make_reports_from_predictions.py
# Entrée attendue :
#   predictions_best.csv (colonnes : Rating/TrueRating, PredictedRating, P1..P5 optionnel)
# Sorties :
#   reports/*.png, reports/*.txt, reports/metrics_binary.json
# ------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend non interactif (utile sur serveurs/CI)
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score,
    f1_score
)

# -----------------------------
# 0) Config I/O
# -----------------------------
PRED_PATH = Path("reports/predictions_best.csv")   # adapter le nom/chemin si besoin
OUT_DIR   = Path("reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# 1) Chargement + vérifications
# -----------------------------
if not PRED_PATH.exists():
    raise FileNotFoundError(f"Fichier introuvable : {PRED_PATH.resolve()}")

df = pd.read_csv(PRED_PATH)
df.columns = [c.strip() for c in df.columns]

rating_col = "Rating" if "Rating" in df.columns else ("TrueRating" if "TrueRating" in df.columns else None)
pred_col   = "PredictedRating" if "PredictedRating" in df.columns else None

if rating_col is None or pred_col is None:
    raise ValueError("Colonnes nécessaires absentes. Il faut au minimum : 'Rating' (ou 'TrueRating') et 'PredictedRating'.")

# Nettoyage types
df = df.dropna(subset=[rating_col, pred_col]).copy()
df[rating_col] = df[rating_col].astype(int)
df[pred_col]   = df[pred_col].astype(int)

# Colonnes probas (optionnelles)
prob_cols = [c for c in df.columns if c.startswith("P") and c[1:].isdigit()]  # P1..P5
prob_cols = sorted(prob_cols, key=lambda c: int(c[1:]))  # ordre croissant

# -----------------------------
# 2) Multiclasse 1..5
# -----------------------------
labels = [1, 2, 3, 4, 5]
y_true = df[rating_col].values
y_pred = df[pred_col].values

# CM brute
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, values_format='d')
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_multiclass.png")
plt.close(fig)

# CM normalisée (par ligne)
cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm_norm, display_labels=labels).plot(ax=ax, values_format='.2f')
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_multiclass_normalized.png")
plt.close(fig)

# Rapport texte (précision, rappel, F1 par classe)
report_txt = classification_report(y_true, y_pred, labels=labels, digits=3, zero_division=0)
with open(OUT_DIR / "classification_report_multiclass.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)

# -----------------------------
# 3) Passage en binaire (1–2=0 ; 3–4–5=1)
# -----------------------------
def to_binary(y):
    return np.array([0 if int(v) in (1, 2) else 1 for v in y])

y_true_bin = to_binary(y_true)
y_pred_bin_hard = to_binary(y_pred)

# Proba d'être "bon" si P1..P5 présents : P3 + P4 + P5
if len(prob_cols) >= 3:
    # On mappe P1..P5 aux classes entières si possible
    class_map = {int(c[1:]): c for c in prob_cols if c[1:].isdigit()}
    pos_probs = np.zeros(len(df), dtype=float)
    for c in (3, 4, 5):
        if c in class_map:
            pos_probs += df[class_map[c]].astype(float).values
else:
    # Sinon, fallback proba dure depuis la prédiction argmax
    pos_probs = np.array([1.0 if p in (3, 4, 5) else 0.0 for p in y_pred], dtype=float)

# -----------------------------
# 4) Métriques binaires + courbes + seuil optimal
# -----------------------------
# Confusion @ seuil 0.5
y_pred_bin_05 = (pos_probs >= 0.5).astype(int)
cm_bin_05 = confusion_matrix(y_true_bin, y_pred_bin_05, labels=[0, 1])
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm_bin_05, display_labels=[0, 1]).plot(ax=ax, values_format='d')
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_binary_0.5.png")
plt.close(fig)

# Seuil optimal par F1 (sur les données fournies)
prec, rec, thr = precision_recall_curve(y_true_bin, pos_probs)
f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
best_idx = int(np.nanargmax(f1s))
best_thr = float(thr[max(0, best_idx - 1)]) if len(thr) > 0 else 0.5

# Confusion @ seuil optimal
y_pred_bin_opt = (pos_probs >= best_thr).astype(int)
cm_bin_opt = confusion_matrix(y_true_bin, y_pred_bin_opt, labels=[0, 1])
fig, ax = plt.subplots()
ConfusionMatrixDisplay(cm_bin_opt, display_labels=[0, 1]).plot(ax=ax, values_format='d')
plt.tight_layout()
plt.savefig(OUT_DIR / "confusion_matrix_binary_opt.png")
plt.close(fig)

# ROC
fpr, tpr, _ = roc_curve(y_true_bin, pos_probs)
roc_auc = auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
ax.plot([0, 1], [0, 1], linestyle='--')
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve (binary)")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig(OUT_DIR / "roc_curve_binary.png")
plt.close(fig)

# Precision–Recall
ap = average_precision_score(y_true_bin, pos_probs)
fig, ax = plt.subplots()
ax.plot(rec, prec, label=f"AP={ap:.3f}")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision–Recall Curve (binary)")
ax.legend(loc="lower left")
plt.tight_layout()
plt.savefig(OUT_DIR / "pr_curve_binary.png")
plt.close(fig)

# JSON de métriques
metrics = {
    "threshold_default_0.5": {
        "f1": float(f1_score(y_true_bin, y_pred_bin_05)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(ap)
    },
    "threshold_optimal": {
        "threshold": best_thr,
        "f1": float(f1_score(y_true_bin, y_pred_bin_opt)),
        "roc_auc": float(roc_auc),
        "pr_auc": float(ap)
    }
}
with open(OUT_DIR / "metrics_binary.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

# -----------------------------
# 5) Récap console
# -----------------------------
print("== Sorties générées dans :", OUT_DIR.resolve())
print("Multiclasse :")
print(" - confusion_matrix_multiclass.png")
print(" - confusion_matrix_multiclass_normalized.png")
print(" - classification_report_multiclass.txt")

print("\nBinaire :")
print(" - confusion_matrix_binary_0.5.png")
print(" - confusion_matrix_binary_opt.png")
print(" - roc_curve_binary.png")
print(" - pr_curve_binary.png")
print(" - metrics_binary.json")

print("\nSeuil optimal (F1) :", round(metrics['threshold_optimal']['threshold'], 3))
print("F1 @ 0.5          :", round(metrics['threshold_default_0.5']['f1'], 3))
print("F1 @ optimal      :", round(metrics['threshold_optimal']['f1'], 3))
print("ROC-AUC           :", round(metrics['threshold_default_0.5']['roc_auc'], 3))
print("PR-AUC            :", round(metrics['threshold_default_0.5']['pr_auc'], 3))
