from datetime import datetime
import os

from airflow import DAG
from airflow.operators.bash import BashOperator


with DAG(
    dag_id="trustpilot_training_pipeline",
    description="Pipeline Airflow pour entraîner le modèle Trustpilot et logger les résultats dans MLflow",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    tags=["trustpilot", "mlops", "mlflow"],
) as dag:

    run_training_mlflow = BashOperator(
        task_id="run_training_mlflow",
        bash_command=(
            "cd /opt/airflow/project && "
            "python training_mlflow.py"
        ),
        env={
            "MLFLOW_TRACKING_URI": "http://mlflow:5000",
            "GIT_COMMIT_HASH": os.getenv("GIT_COMMIT_HASH", "unknown"),
        },
        append_env=True,
)