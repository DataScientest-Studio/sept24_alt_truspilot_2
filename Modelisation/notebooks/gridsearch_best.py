import argparse
import json
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


# Grille d'hyperparamètres :
# - Côté TF-IDF :
#     * max_features : taille du vocabulaire (plus grand = plus de mots/expressions conservés)
#     * ngram_range : 1 gramme (mots seuls), 2 grammes (bigrams), 3 grammes (trigrams)
#     * min_df      : ignore les termes trop rares (bruit)
# - Côté LogReg :
#     * C (inverse de la régularisation, plus petit = régularisation plus forte)
#     * class_weight : peut aider si pas d'oversampling ; ici souvent None suffit (on oversample déjà)
PARAM_GRID = {
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__penalty": ["l2"],
    "clf__class_weight": [None, "balanced"],
}


def main(train_csv: str, test_csv: str, outdir: str, cv: int = 3, n_jobs: int = -1, seed: int = 42) -> None:
    """Optimise une Régression Logistique en maximisant F1-macro via GridSearchCV.

    Pourquoi LogReg ?
      - Très performante sur données texte TF-IDF
      - Interprétable et rapide
      - Supporte predict_proba (utile pour applications aval)
    """
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Lecture des jeux de données
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    X_train, y_train = train["CleanText"].astype(str), train["Rating"].astype(int)
    X_test, y_test = test["CleanText"].astype(str), test["Rating"].astype(int)

    # 2) Pipeline = TF-IDF → Oversampling → LogReg
    pipe = ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("ros", RandomOverSampler(random_state=seed)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
    ])

    # 3) GridSearchCV : on évalue des combinaisons d'hyperparams via validation croisée
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        cv=cv,                  # ex: 3 folds
        scoring="f1_macro",     # métrique pivot
        n_jobs=n_jobs,          # parallélisation
        verbose=1,              # affiche la progression
    )

    # 4) Entraînement (fit) → le meilleur pipeline complet est accessible via best_estimator_
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # 5) Évaluation finale sur le jeu de test (jamais vu pendant le fit)
    y_pred = best.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    # 6) Sauvegardes (fichiers faciles à montrer en soutenance)
    with open(out / "best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, digits=3))

    try:
        import joblib
        joblib.dump(best, out / "best_model.joblib")  # pipeline complet (vectorizer + sampler + clf)
    except Exception as e:
        print("⚠️ Sauvegarde joblib échouée :", e)

    # 7) Récap console
    print("✅ Meilleurs hyperparamètres :")
    print(grid.best_params_)
    print("\n📊 Scores sur le jeu de test :")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GridSearchCV LogReg sur TF-IDF avec oversampling")
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--outdir", type=str, default="models/gridsearch_logreg")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(args.train, args.test, args.outdir, cv=args.cv, n_jobs=args.n_jobs, seed=args.seed)
