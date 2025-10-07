import argparse
from pathlib import Path
import pandas as pd
import joblib
import imblearn  # nécessaire si pipeline imblearn.Pipeline
import numpy as np
from textblob import TextBlob 
from typing import Optional

CANDIDATE_MODELS = [
    # ordre de préférence : meilleur d'abord
    ("models/sentiment_logreg_paramsTF_IDF/best_model_sentiment.joblib", "TF-IDF + Sentiment + Tweaks"),
    ("models/sentiment_logreg/best_model_sentiment.joblib", "TF-IDF + Sentiment"),
    ("models/gridsearch_logreg/best_model.joblib", "GridSearch LogReg"),
    ("models/baseline/best_baseline.joblib", "Baseline"),
]

def resolve_model_path(cli_path: Optional[str]) -> Path:
    """Trouve le modèle à charger : --model sinon meilleurs emplacements connus."""
    if cli_path:
        p = Path(cli_path)
        return p if p.is_absolute() else (Path.cwd() / p)

    # notebooks/.. = racine projet
    ROOT = Path(__file__).resolve().parents[1]
    for rel, label in CANDIDATE_MODELS:
        p = ROOT / rel
        if p.exists():
            print(f"🔎 Modèle détecté : {label} → {p}")
            return p
    raise FileNotFoundError(
        "Aucun modèle trouvé aux emplacements standards.\n"
        + "\n".join(f"- {rel}" for rel,_ in CANDIDATE_MODELS)
        + "\nOu passez --model <chemin_vers_joblib>."
    )


import numpy as np
from textblob import TextBlob

def get_sentiment(texts):
    
    return np.array([TextBlob(t).sentiment.polarity for t in texts]).reshape(-1, 1)


def predict_csv(pipe, input_csv: Path, text_col: str, output_csv: Path, with_proba: bool):
    print(f"📥 Lecture: {input_csv}")
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise KeyError(f"Colonne '{text_col}' absente. Colonnes: {list(df.columns)}")

    X = df[text_col].astype(str)
    print("🔮 Prédictions en cours…")
    df["PredictedRating"] = pipe.predict(X)

    if with_proba:
        # Certaines classes peuvent être ordonnées différemment selon le modèle ; alignons sur [1..5]
        try:
            proba = pipe.predict_proba(X)
            # récupère l'ordre des classes du modèle
            classes = list(getattr(pipe, "classes_", getattr(pipe[-1], "classes_", [])))
            if classes and len(classes) == proba.shape[1]:
                # map vers colonnes P1..P5 dans l'ordre 1..5
                class_to_idx = {c: i for i, c in enumerate(classes)}
                for c in [1, 2, 3, 4, 5]:
                    if c in class_to_idx:
                        df[f"P{c}"] = proba[:, class_to_idx[c]]
            else:
                # fallback: on sort tel quel
                for j in range(proba.shape[1]):
                    df[f"P{j}"] = proba[:, j]
        except Exception:
            print("ℹ️ Modèle sans predict_proba (ex: LinearSVC). Probabilités non ajoutées.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Écrit: {output_csv} ({len(df)} lignes)")

def demo(pipe):
    tests = [
        "Amazing visit, friendly staff and quick support. Highly recommend!",
        "Terrible experience. Item never arrived and customer service was rude.",
        "Service correct mais livraison un peu lente, globalement satisfait."
    ]
    print("🔎 Démo prédictions:", pipe.predict(tests))
    # Probabilités si dispo
    try:
        proba = pipe.predict_proba(tests)
        print("🔎 Probabilités (si dispo):")
        print(proba)
    except Exception:
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de notes (1..5) avec pipeline .joblib")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le .joblib (optionnel)")
    parser.add_argument("--input_csv", type=str, default=None, help="CSV d'entrée (optionnel)")
    parser.add_argument("--text_col", type=str, default="CleanText", help="Colonne texte (défaut: CleanText)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="CSV de sortie (défaut: predictions.csv)")
    parser.add_argument("--proba", action="store_true", help="Ajoute les probabilités P1..P5 si disponibles")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"🧠 Chargement modèle: {model_path}")
    pipe = joblib.load(model_path)

    if args.input_csv:
        in_path = Path(args.input_csv)
        out_path = Path(args.output_csv)
        if not in_path.is_absolute():
            in_path = Path.cwd() / in_path
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        predict_csv(pipe, in_path, args.text_col, out_path, with_proba=args.proba)
    else:
        demo(pipe)
