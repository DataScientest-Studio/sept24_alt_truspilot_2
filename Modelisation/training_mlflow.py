from pathlib import Path
import hashlib
import sqlite3

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DB_PATH = "data/trustpilot.db"
TABLE_NAME = "reviews"
MODEL_PATH = "models/trustpilot_logistic_tfidf.joblib"

EXPERIMENT_NAME = "trustpilot_sentiment_experiment"
REGISTERED_MODEL_NAME = "trustpilot_sentiment_model"
BEST_ALIAS = "best"
METRIC_TO_COMPARE = "f1_macro"


def load_data() -> pd.DataFrame:
    if not Path(DB_PATH).exists():
        raise FileNotFoundError(f"Base de données introuvable : {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    query = f"SELECT CleanText, Rating FROM {TABLE_NAME}"
    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["CleanText", "Rating"]).copy()
    df["target"] = (df["Rating"] >= 4).astype(int)
    return df


def compute_dataset_hash(df: pd.DataFrame) -> str:
    """
    Calcule une empreinte simple du dataset utilisé pour l'entraînement.
    Cela permet d'identifier si les données ont changé entre deux runs.
    """
    df_hash = pd.util.hash_pandas_object(df, index=True).values
    return hashlib.md5(df_hash).hexdigest()


def build_pipeline() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    max_features=5000,
                    ngram_range=(1, 2),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )


def get_best_model_metric(
    client: MlflowClient,
    model_name: str,
    alias: str,
    metric_name: str,
):
    """
    Récupère la métrique du modèle actuellement marqué comme 'best'.
    Si aucun modèle best n'existe encore, retourne None.
    """
    try:
        best_version = client.get_model_version_by_alias(model_name, alias)
        run = client.get_run(best_version.run_id)
        return run.data.metrics.get(metric_name), best_version.version

    except Exception:
        return None, None


def set_model_as_best(client: MlflowClient, model_name: str, version: str):
    """
    Marque une version du modèle comme meilleure version grâce à un alias MLflow.
    """
    client.set_registered_model_alias(
        name=model_name,
        alias=BEST_ALIAS,
        version=version,
    )


def log_confusion_matrix_to_mlflow(cm):
    """
    Logge la matrice de confusion dans MLflow :
    - les 4 valeurs comme métriques scalaires ;
    - une image PNG comme artifact.
    """
    tn, fp, fn, tp = cm.ravel()

    mlflow.log_metric("confusion_true_negative", int(tn))
    mlflow.log_metric("confusion_false_positive", int(fp))
    mlflow.log_metric("confusion_false_negative", int(fn))
    mlflow.log_metric("confusion_true_positive", int(tp))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["negative", "positive"],
    )

    disp.plot(values_format="d")
    plt.title("Confusion Matrix - Trustpilot Sentiment Model")
    plt.tight_layout()

    mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
    plt.close()


def train():
    Path("models").mkdir(exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    df = load_data()
    df = prepare_data(df)

    dataset_hash = compute_dataset_hash(df)
    target_distribution = df["target"].value_counts().to_dict()

    X = df["CleanText"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro")
    precision_macro = precision_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )
    recall_macro = recall_score(
        y_test,
        y_pred,
        average="macro",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("Informations dataset :")
    print(f"Dataset hash : {dataset_hash}")
    print(f"Dataset size : {len(df)}")
    print(f"Distribution target : {target_distribution}")
    print()
    print("Matrice de confusion :")
    print(cm)
    print()
    print("Rapport de classification :")
    print(report)

    with mlflow.start_run(run_name="tfidf_logistic_regression_training") as run:
        # Paramètres du modèle
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("target_definition", "1 if Rating >= 4 else 0")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("ngram_range", "(1, 2)")
        mlflow.log_param("class_weight", "balanced")

        # Informations sur le split
        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))

        # Versioning simple des données
        mlflow.log_param("source_database_path", DB_PATH)
        mlflow.log_param("source_table_name", TABLE_NAME)
        mlflow.log_param("dataset_hash", dataset_hash)
        mlflow.log_param("dataset_size", len(df))
        mlflow.log_param("negative_class_count", int(target_distribution.get(0, 0)))
        mlflow.log_param("positive_class_count", int(target_distribution.get(1, 0)))

        # Métriques principales
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_macro", precision_macro)
        mlflow.log_metric("recall_macro", recall_macro)

        # Matrice de confusion
        log_confusion_matrix_to_mlflow(cm)

        # Enregistrement du modèle dans MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        # Sauvegarde locale pour l'API FastAPI
        joblib.dump(pipeline, MODEL_PATH)

        latest_versions = client.search_model_versions(
            f"name='{REGISTERED_MODEL_NAME}'"
        )

        current_version = max(
            latest_versions,
            key=lambda version: int(version.version),
        )

        previous_best_metric, previous_best_version = get_best_model_metric(
            client=client,
            model_name=REGISTERED_MODEL_NAME,
            alias=BEST_ALIAS,
            metric_name=METRIC_TO_COMPARE,
        )

        if previous_best_metric is None:
            set_model_as_best(
                client=client,
                model_name=REGISTERED_MODEL_NAME,
                version=current_version.version,
            )

            best_decision = "Premier modèle enregistré : marqué comme best."

        elif f1_macro > previous_best_metric:
            set_model_as_best(
                client=client,
                model_name=REGISTERED_MODEL_NAME,
                version=current_version.version,
            )

            best_decision = (
                f"Nouveau modèle meilleur que la version "
                f"{previous_best_version} : "
                f"{f1_macro:.4f} > {previous_best_metric:.4f}. "
                f"Version {current_version.version} marquée comme best."
            )

        else:
            best_decision = (
                f"Nouveau modèle non retenu. "
                f"Score actuel : {f1_macro:.4f}, "
                f"meilleur score précédent : {previous_best_metric:.4f}, "
                f"version {previous_best_version} conservée comme best."
            )

        mlflow.log_param("best_model_decision", best_decision)

        print()
        print("Entraînement terminé.")
        print(f"Run ID : {run.info.run_id}")
        print(f"Dataset hash : {dataset_hash}")
        print(f"Dataset size : {len(df)}")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"F1 macro : {f1_macro:.4f}")
        print(f"Precision macro : {precision_macro:.4f}")
        print(f"Recall macro : {recall_macro:.4f}")
        print(f"Model Registry : {REGISTERED_MODEL_NAME}")
        print(f"Version créée : {current_version.version}")
        print(best_decision)


if __name__ == "__main__":
    train()