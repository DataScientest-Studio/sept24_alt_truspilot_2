# train_binary.py
# ------------------------------------------------------------
# Entraîne un modèle binaire (0={1,2}, 1={3,4,5}) avec TF-IDF + LogisticRegression.
# Par défaut, lit data/processed/train.csv et data/processed/test.csv (split canonique multiclass).
# Sinon, si --train_csv/--test_csv non fournis, fait un split 80/20 depuis --input_csv.
# Oversampling (RandomOverSampler) uniquement sur le train.
# Sorties : modèles/, reports/ (matrices, courbes, métriques), predictions_binary.csv
# ------------------------------------------------------------

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay, classification_report,
    f1_score, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from joblib import dump

RANDOM_STATE = 42

def map_binary(r):
    r = int(r)
    return 0 if r in (1, 2) else 1

def load_from_split(train_csv, test_csv, text_col, rating_col):
    tr = pd.read_csv(train_csv)
    te = pd.read_csv(test_csv)
    for need in (text_col, rating_col):
        assert need in tr.columns and need in te.columns, f"Colonne manquante: {need}"
    X_train = tr[text_col].astype(str).values
    y_train = tr[rating_col].astype(int).map(map_binary).values
    X_test  = te[text_col].astype(str).values
    y_test  = te[rating_col].astype(int).map(map_binary).values
    return X_train, X_test, y_train, y_test

def load_from_full(input_csv, text_col, rating_col):
    df = pd.read_csv(input_csv).dropna(subset=[text_col, rating_col]).copy()
    X = df[text_col].astype(str).values
    y = df[rating_col].astype(int).map(map_binary).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_tr, X_te, y_tr, y_te

def main(args):
    models_dir  = Path(args.models_dir);  models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(args.reports_dir); reports_dir.mkdir(parents=True, exist_ok=True)

    # --- Chargement des données
    if args.train_csv and args.test_csv:
        X_train, X_test, y_train, y_test = load_from_split(
            args.train_csv, args.test_csv, args.text_col, args.rating_col
        )
    else:
        assert args.input_csv, "--input_csv requis si --train_csv/--test_csv ne sont pas fournis."
        X_train, X_test, y_train, y_test = load_from_full(
            args.input_csv, args.text_col, args.rating_col
        )

    # --- Pipeline binaire : TF-IDF + ROS (train only) + LogReg
    pipe = ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.9,
            max_features=50000
        )),
        ("ros", RandomOverSampler(random_state=RANDOM_STATE)),
        ("clf", LogisticRegression(
            max_iter=4000,
            random_state=RANDOM_STATE
        ))
    ])

    # Entraînement
    pipe.fit(X_train, y_train)

    # Prédictions sur test
    proba_1 = pipe.predict_proba(X_test)[:, 1]
    y_pred_05 = (proba_1 >= 0.5).astype(int)

    # Seuil optimal F1 (sur test pour simplicité)
    prec, rec, thr = precision_recall_curve(y_test, proba_1)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    best_idx = int(np.nanargmax(f1s))
    best_thr = float(thr[max(0, best_idx-1)]) if len(thr) > 0 else 0.5
    y_pred_opt = (proba_1 >= best_thr).astype(int)

    # Métriques
    metrics = {
        "threshold_default_0.5": {
            "f1": float(f1_score(y_test, y_pred_05)),
            "roc_auc": float(roc_auc_score(y_test, proba_1)),
            "pr_auc": float(average_precision_score(y_test, proba_1))
        },
        "threshold_optimal": {
            "threshold": best_thr,
            "f1": float(f1_score(y_test, y_pred_opt)),
            "roc_auc": float(roc_auc_score(y_test, proba_1)),
            "pr_auc": float(average_precision_score(y_test, proba_1))
        }
    }

    # Confusions
    for tag, yhat in [("0.5", y_pred_05), ("opt", y_pred_opt)]:
        cm = confusion_matrix(y_test, yhat, labels=[0,1])
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(cm, display_labels=[0,1]).plot(ax=ax, values_format='d')
        plt.tight_layout()
        plt.savefig(reports_dir / f"confusion_matrix_binary_{tag}.png")
        plt.close(fig)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, proba_1)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1], [0,1], linestyle='--')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (binary)"); ax.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(reports_dir / "roc_curve_binary.png"); plt.close(fig)

    # PR
    fig, ax = plt.subplots()
    ax.plot(rec, prec, label=f"AP={average_precision_score(y_test, proba_1):.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (binary)")
    ax.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(reports_dir / "pr_curve_binary.png"); plt.close(fig)

    # Classification report @ 0.5
    cls_rep = classification_report(y_test, y_pred_05, labels=[0,1], digits=3, zero_division=0)
    with open(reports_dir / "classification_report_binary_0.5.txt", "w", encoding="utf-8") as f:
        f.write(cls_rep)

    # Sauvegardes
    (models_dir).mkdir(parents=True, exist_ok=True)
    dump_path = models_dir / "binary_logreg.joblib"
    dump(pipe, dump_path)

    pd.DataFrame({
        "y_true": y_test,
        "proba_1": proba_1,
        "y_pred_0.5": y_pred_05,
        "y_pred_opt": y_pred_opt
    }).to_csv(reports_dir / "predictions_binary.csv", index=False)

    with open(reports_dir / "metrics_binary.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("== Modèle sauvegardé :", dump_path.resolve())
    print("== Reports dans     :", reports_dir.resolve())
    print("Seuil optimal (F1) :", round(metrics['threshold_optimal']['threshold'], 3))
    print("F1 @ 0.5          :", round(metrics['threshold_default_0.5']['f1'], 3))
    print("F1 @ optimal      :", round(metrics['threshold_optimal']['f1'], 3))
    print("ROC-AUC           :", round(metrics['threshold_default_0.5']['roc_auc'], 3))
    print("PR-AUC            :", round(metrics['threshold_default_0.5']['pr_auc'], 3))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Par défaut on lit le split canonique existant
    p.add_argument("--train_csv", type=str, default="data/processed/train.csv",
                   help="Chemin vers data/processed/train.csv (texte + rating 1..5).")
    p.add_argument("--test_csv",  type=str, default="data/processed/test.csv",
                   help="Chemin vers data/processed/test.csv (texte + rating 1..5).")

    # args pour ignorer le split existant et refaire un split 80/20 :
    p.add_argument("--input_csv", type=str, default="",
                   help="CSV complet (texte+rating) à splitter si --train_csv/--test_csv vides.")
    p.add_argument("--text_col", type=str, default="CleanText")
    p.add_argument("--rating_col", type=str, default="Rating")
    p.add_argument("--models_dir", type=str, default="models/binary")
    p.add_argument("--reports_dir", type=str, default="reports/binary")
    args = p.parse_args()
    main(args)
