import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


# =====================
# MODELES
# =====================
MODELS = {
    "logreg": LogisticRegression(max_iter=1000, random_state=42),
    "linearsvc": LinearSVC(random_state=42),
    "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
}


def build_pipeline(estimator):
    """Construit le pipeline complet : TF-IDF + RandomOverSampler + Modèle"""
    return ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", estimator)
    ])


def evaluate(y_true, y_pred):
    """Calcule les métriques principales"""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def save_confusion_matrix(y_true, y_pred, out_png: Path, title: str):
    """Génère et sauvegarde la matrice de confusion"""
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(5)
    plt.xticks(tick_marks, [1, 2, 3, 4, 5])
    plt.yticks(tick_marks, [1, 2, 3, 4, 5])
    plt.ylabel('Vrai label')
    plt.xlabel('Label prédit')
    plt.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main(train_csv: str, test_csv: str, outdir: str):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train, y_train = train["CleanText"].astype(str), train["Rating"].astype(int)
    X_test, y_test = test["CleanText"].astype(str), test["Rating"].astype(int)

    rows = []
    best_model_name, best_f1 = None, -1

    for name, est in MODELS.items():
        print(f"\n===== Entraînement modèle : {name} =====")
        pipe = build_pipeline(est)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        metrics = evaluate(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        # Sauvegarde résultats
        (outdir / name).mkdir(exist_ok=True)
        with open(outdir / name / "classification_report.txt", "w") as f:
            f.write(report)
        with open(outdir / name / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        save_confusion_matrix(y_test, y_pred, outdir / name / "confusion_matrix.png", f"CM — {name}")

        rows.append({"model": name, **metrics})

        # Garde le meilleur modèle selon f1_macro
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model_name = name
            try:
                import joblib
                joblib.dump(pipe, outdir / "best_baseline.joblib")
            except Exception as e:
                print("⚠️ Erreur sauvegarde modèle :", e)

    # Résumé global
    df_results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df_results.to_csv(outdir / "baseline_metrics.csv", index=False)
    print("\nRésultats comparatifs :")
    print(df_results)
    print(f"\n🏆 Meilleur modèle : {best_model_name} — F1_macro = {best_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--outdir", type=str, default="models/baseline")
    args = parser.parse_args()
    main(args.train, args.test, args.outdir)
