import argparse
import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


# Pourquoi ces 3 modèles ?
# - LogisticRegression : linéaire, rapide, souvent excellente avec TF-IDF
# - LinearSVC        : SVM linéaire, robuste sur textes, mais pas de predict_proba
# - RandomForest     : non linéaire, bon benchmark mais peut favoriser la classe majoritaire
MODELS: Dict[str, Any] = {
    "logreg": LogisticRegression(max_iter=1000, random_state=42),
    "linearsvc": LinearSVC(random_state=42),
    "rf": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
}


def build_pipeline(estimator):
    """Pipeline complet :
    1) TF-IDF (texte → vecteurs numériques)
    2) RandomOverSampler (équilibrage du TRAIN en dupliquant les classes rares)
    3) Modèle de classification (LogReg / LinearSVC / RF)

    ⚠️ L'oversampling est dans le pipeline pour être appliqué **uniquement pendant fit()**.
       Sur le jeu de test (predict), **aucun** rééchantillonnage n'est appliqué → évaluation honnête.
    """
    return ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", estimator),
    ])


def evaluate(y_true, y_pred) -> Dict[str, float]:
    """Métriques principales :
    - accuracy : % de bonnes prédictions (peut être trompeur si classes déséquilibrées)
    - balanced_accuracy : moyenne des rappels par classe
    - f1_macro : moyenne des F1 par classe (métrique pivot ici)
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
    }


def save_confusion_matrix(y_true, y_pred, out_png: Path, title: str) -> None:
    """Génère une matrice de confusion simple et l'enregistre en PNG.
    Astuce lecture :
      - Lignes = vraies classes (1..5)
      - Colonnes = classes prédites
      - Plus la diagonale est foncée, mieux c'est.
    """
    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')  # palette lisible
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


def main(train_csv: str, test_csv: str, outdir: str) -> None:
    """Entraîne les 3 modèles, compare leurs scores et sauvegarde :
      - un CSV récapitulatif des métriques
      - un rapport de classification + matrice de confusion par modèle
      - le **meilleur pipeline** au format joblib (prêt à être rechargé pour faire des prédictions)
    """
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    # 1) Lecture des jeux de données créés par le split
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train, y_train = train["CleanText"].astype(str), train["Rating"].astype(int)
    X_test, y_test = test["CleanText"].astype(str), test["Rating"].astype(int)

    rows = []
    best_model_name, best_f1 = None, -1.0

    # 2) Boucle sur les 3 modèles pour entraîner/évaluer
    for name, est in MODELS.items():
        print(f"\n===== Entraînement modèle : {name} =====")
        pipe = build_pipeline(est)
        pipe.fit(X_train, y_train)      # ← TF-IDF + Oversampling appliqués ici
        y_pred = pipe.predict(X_test)   # ← uniquement TF-IDF + CLF (pas d'oversampling sur test)

        # 3) Évalue et sauvegarde les résultats
        metrics = evaluate(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        (outdir_path / name).mkdir(exist_ok=True)
        with open(outdir_path / name / "classification_report.txt", "w") as f:
            f.write(report)
        with open(outdir_path / name / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        save_confusion_matrix(y_test, y_pred, outdir_path / name / "confusion_matrix.png", f"CM — {name}")

        rows.append({"model": name, **metrics})

        # 4) On retient le meilleur selon F1-macro (métrique pivot)
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model_name = name
            try:
                import joblib
                joblib.dump(pipe, outdir_path / "best_baseline.joblib")
            except Exception as e:
                print("⚠️ Erreur sauvegarde modèle :", e)

    # 5) Tableau de synthèse comparatif
    df_results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    df_results.to_csv(outdir_path / "baseline_metrics.csv", index=False)
    print("\nRésultats comparatifs :")
    print(df_results)
    print(f"\n🏆 Meilleur modèle : {best_model_name} — F1_macro = {best_f1:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baselines TF-IDF + Oversampling + 3 modèles")
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--outdir", type=str, default="models/baseline")
    args = parser.parse_args()

    main(args.train, args.test, args.outdir)
