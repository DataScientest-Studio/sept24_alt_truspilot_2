import argparse
import json
from pathlib import Path
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


# Grille simple, efficace et lisible pour la soutenance
PARAM_GRID = {
    "tfidf__max_features": [5000, 10000, 20000],
    "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
    "tfidf__min_df": [1, 2, 3],
    "clf__C": [0.5, 1.0, 2.0, 4.0],
    "clf__penalty": ["l2"],
    "clf__class_weight": [None, "balanced"],
}


def main(train_csv: str, test_csv: str, outdir: str, cv: int = 3, n_jobs: int = -1, seed: int = 42):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train, y_train = train["CleanText"].astype(str), train["Rating"].astype(int)
    X_test, y_test = test["CleanText"].astype(str), test["Rating"].astype(int)

    pipe = ImbPipeline(steps=[
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("ros", RandomOverSampler(random_state=seed)),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed))
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=PARAM_GRID,
        cv=cv,
        scoring="f1_macro",
        n_jobs=n_jobs,
        verbose=1,
    )

    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    y_pred = best.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    # Sauvegardes
    with open(out / "best_params.json", "w") as f:
        json.dump(grid.best_params_, f, indent=2)
    with open(out / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(out / "classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred, digits=3))

    try:
        import joblib
        joblib.dump(best, out / "best_model.joblib")
    except Exception as e:
        print("⚠️ Sauvegarde joblib échouée :", e)

    print("✅ Meilleurs hyperparamètres :")
    print(grid.best_params_)
    print("\n📊 Scores sur le jeu de test :")
    print(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--outdir", type=str, default="models/gridsearch_logreg")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.train, args.test, args.outdir, cv=args.cv, n_jobs=args.n_jobs, seed=args.seed)
