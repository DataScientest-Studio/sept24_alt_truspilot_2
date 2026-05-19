import sqlite3
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DB_PATH = "data/trustpilot.db"
TABLE_NAME = "reviews"
MODEL_PATH = "models/trustpilot_logistic_tfidf.joblib"


def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT CleanText, Rating FROM {TABLE_NAME}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["CleanText", "Rating"]).copy()
    df["target"] = (df["Rating"] >= 4).astype(int)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])


def train() -> None:
    Path("models").mkdir(exist_ok=True)

    df = load_data()
    df = prepare_target(df)

    X = df["CleanText"]
    y = df["target"]

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    joblib.dump(pipeline, MODEL_PATH)

    print("Entraînement terminé avec succès.")
    print(f"Modèle sauvegardé ici : {MODEL_PATH}")
    print(f"Nombre de lignes utilisées : {len(df)}")
    print("Répartition de la cible :")
    print(df["target"].value_counts().to_dict())


if __name__ == "__main__":
    train()