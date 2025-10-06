import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
from textblob import TextBlob

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler


# -----------------------------------------------------------
# Fonction utilitaire : calcul du sentiment TextBlob
# -----------------------------------------------------------
def get_sentiment(texts):
    return np.array([TextBlob(t).sentiment.polarity for t in texts]).reshape(-1, 1)


# -----------------------------------------------------------
# Pipeline complet : TF-IDF + sentiment
# -----------------------------------------------------------
def build_pipeline():
    # Partie texte → TF-IDF
    tfidf = ("tfidf", TfidfVectorizer(max_features=10000, ngram_range=(1, 2)))

    # Partie sentiment → transforme X (texte brut) en score de sentiment
    sentiment = ("sentiment", FunctionTransformer(get_sentiment, validate=False))

    # Fusion TF-IDF + Sentiment
    features = FeatureUnion([
        tfidf,
        sentiment
    ])

    # Oversampling + modèle
    pipe = ImbPipeline([
        ("features", features),
        ("ros", RandomOverSampler(random_state=42)),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])

    return pipe


# -----------------------------------------------------------
# Entraînement & évaluation
# -----------------------------------------------------------
def main(train_csv, test_csv, outdir):
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    X_train, y_train = train["CleanText"].astype(str), train["Rating"].astype(int)
    X_test, y_test = test["CleanText"].astype(str), test["Rating"].astype(int)

    print("🧠 Construction du pipeline TF-IDF + sentiment + LogReg ...")
    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    print("🔮 Prédiction sur le test ...")
    y_pred = pipe.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "f1_macro": f1_score(y_test, y_pred, average="macro"),
    }

    print("✅ Résultats :")
    print(json.dumps(metrics, indent=2))
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred, digits=3))

    with open(outdir_path / "metrics_sentiment.json", "w") as f:
        json.dump(metrics, f, indent=2)

    try:
        import joblib
        joblib.dump(pipe, outdir_path / "best_model_sentiment.joblib")
        print(f"💾 Modèle sauvegardé : {outdir_path / 'best_model_sentiment.joblib'}")
    except Exception as e:
        print("⚠️ Erreur de sauvegarde :", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TF-IDF + Sentiment Feature + LogReg")
    parser.add_argument("--train", type=str, default="data/processed/train.csv")
    parser.add_argument("--test", type=str, default="data/processed/test.csv")
    parser.add_argument("--outdir", type=str, default="models/sentiment_logreg")
    args = parser.parse_args()

    main(args.train, args.test, args.outdir)
