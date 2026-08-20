from pathlib import Path
import os
import sqlite3
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel, Field
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


DB_PATH = "data/trustpilot.db"
TABLE_NAME = "reviews"
MODEL_PATH = "models/trustpilot_logistic_tfidf.joblib"

API_KEY = os.getenv("API_KEY", "dev-secret-key")


app = FastAPI(
    title="Trustpilot Sentiment API",
    description="API de prédiction de sentiment à partir d'avis clients Trustpilot.",
    version="1.0.0",
)

Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False
)

model = None

# Référence issue du dataset d'entraînement :
# 3652 avis positifs sur 5619 lignes utilisées.
REFERENCE_POSITIVE_RATIO = 3652 / 5619

total_predictions = 0
positive_predictions = 0


prediction_counter = Counter(
    "trustpilot_predictions_total",
    "Nombre total de prédictions réalisées par l'API",
    ["label"]
)

prediction_positive_probability = Histogram(
    "trustpilot_prediction_positive_probability",
    "Distribution des probabilités positives retournées par le modèle",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

input_text_length = Histogram(
    "trustpilot_input_text_length",
    "Longueur des textes envoyés à l'API de prédiction",
    buckets=[0, 50, 100, 200, 500, 1000, 2000, 5000]
)

current_positive_ratio_gauge = Gauge(
    "trustpilot_current_positive_ratio",
    "Ratio courant de prédictions positives depuis le démarrage de l'API"
)

reference_positive_ratio_gauge = Gauge(
    "trustpilot_reference_positive_ratio",
    "Ratio de référence des avis positifs dans le dataset d'entraînement"
)

drift_proxy_gauge = Gauge(
    "trustpilot_prediction_drift_proxy",
    "Indicateur simple de drift basé sur l'écart entre le ratio positif courant et le ratio positif de référence"
)

reference_positive_ratio_gauge.set(REFERENCE_POSITIVE_RATIO)


def verify_api_key(x_api_key: str | None = Header(default=None, alias="X-API-Key")):
    """
    Vérifie la clé API envoyée dans le header X-API-Key.
    Les routes sensibles comme /predict et /training sont protégées.
    """
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte de l'avis client")


class PredictResponse(BaseModel):
    text: str
    prediction: int
    label: str
    probability_negative: Optional[float]
    probability_positive: Optional[float]


@app.on_event("startup")
def startup_event():
    """
    Chargement du modèle au démarrage de l'API.
    Cela évite de recharger le modèle à chaque requête.
    """
    global model

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
    else:
        model = None


def load_model():
    global model

    if model is None:
        if not Path(MODEL_PATH).exists():
            raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)

    return model


def load_data() -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Base de données introuvable : {DB_PATH}")

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


@app.get("/")
def root():
    return {
        "message": "Trustpilot Sentiment API is running",
        "docs_url": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": Path(MODEL_PATH).exists(),
        "database_available": Path(DB_PATH).exists()
    }


@app.get("/model-info")
def model_info():
    return {
        "model_name": "TF-IDF + Logistic Regression",
        "problem_type": "binary classification",
        "input_feature": "CleanText",
        "target_definition": "1 if Rating >= 4 else 0",
        "labels": {
            "0": "negative",
            "1": "positive"
        },
        "model_path": MODEL_PATH
    }


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)]
)
def predict(request: PredictRequest):
    try:
        loaded_model = load_model()

        prediction = int(loaded_model.predict([request.text])[0])
        label = "positive" if prediction == 1 else "negative"

        probability_negative = None
        probability_positive = None

        if hasattr(loaded_model, "predict_proba"):
            probabilities = loaded_model.predict_proba([request.text])[0]
            probability_negative = float(probabilities[0])
            probability_positive = float(probabilities[1])
        
        global total_predictions, positive_predictions

        prediction_counter.labels(label=label).inc()
        input_text_length.observe(len(request.text))

        if probability_positive is not None:
            prediction_positive_probability.observe(probability_positive)

        total_predictions += 1

        if prediction == 1:
            positive_predictions += 1

        current_positive_ratio = positive_predictions / total_predictions
        current_positive_ratio_gauge.set(current_positive_ratio)

        drift_proxy = abs(current_positive_ratio - REFERENCE_POSITIVE_RATIO)
        drift_proxy_gauge.set(drift_proxy)
        
        return {
            "text": request.text,
            "prediction": prediction,
            "label": label,
            "probability_negative": probability_negative,
            "probability_positive": probability_positive,
        }

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant la prédiction : {str(e)}"
        )


@app.post(
    "/training",
    dependencies=[Depends(verify_api_key)]
)
def training():
    """
    Route d'entraînement simple.
    Elle sera ensuite complétée par un vrai script MLflow séparé.
    """
    try:
        Path("models").mkdir(exist_ok=True)

        df = load_data()
        df = prepare_target(df)

        X = df["CleanText"]
        y = df["target"]

        pipeline = build_pipeline()
        pipeline.fit(X, y)

        joblib.dump(pipeline, MODEL_PATH)

        global model
        model = pipeline

        return {
            "message": "Entraînement terminé avec succès.",
            "model_path": MODEL_PATH,
            "rows_used": int(len(df)),
            "target_distribution": {
                str(k): int(v) for k, v in df["target"].value_counts().to_dict().items()
            },
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erreur pendant l'entraînement : {str(e)}"
        )