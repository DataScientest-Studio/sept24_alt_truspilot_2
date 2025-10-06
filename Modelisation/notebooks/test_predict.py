import argparse
from pathlib import Path
import pandas as pd
import joblib
import imblearn  # nécessaire si pipeline imblearn.Pipeline

def resolve_model_path(cli_path: str | None) -> Path:
    if cli_path:
        p = Path(cli_path)
        return p if p.is_absolute() else (Path.cwd() / p)
    # Cherche modèle à partir de la racine projet (notebooks/.. = racine)
    ROOT = Path(__file__).resolve().parents[1]
    candidates = [
        ROOT / "models" / "gridsearch_logreg" / "best_model.joblib",
        ROOT / "models" / "baseline" / "best_baseline.joblib",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Aucun modèle trouvé dans :\n" + "\n".join(map(str, candidates)))

def predict_csv(pipe, input_csv: Path, text_col: str, output_csv: Path):
    print(f"📥 Lecture: {input_csv}")
    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise KeyError(f"Colonne '{text_col}' absente. Colonnes: {list(df.columns)}")
    print("🔮 Prédictions en cours…")
    df["PredictedRating"] = pipe.predict(df[text_col].astype(str))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Écrit: {output_csv} ({len(df)} lignes)")

def demo(pipe):
    tests = [
        "Amazing visit, friendly staff and quick support. Highly recommend!",
        "Terrible experience. Item never arrived and customer service was rude."
    ]
    print("🔎 Démo prédictions:", pipe.predict(tests))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prédiction de notes (1..5) avec pipeline .joblib")
    parser.add_argument("--model", type=str, default=None, help="Chemin vers le .joblib (optionnel)")
    parser.add_argument("--input_csv", type=str, default=None, help="CSV d'entrée (optionnel)")
    parser.add_argument("--text_col", type=str, default="CleanText", help="Colonne texte (défaut: CleanText)")
    parser.add_argument("--output_csv", type=str, default="predictions.csv", help="CSV de sortie (défaut: predictions.csv)")
    args = parser.parse_args()

    model_path = resolve_model_path(args.model)
    print(f"🧠 Chargement modèle: {model_path}")
    pipe = joblib.load(model_path)

    if args.input_csv:
        in_path = Path(args.input_csv)
        out_path = Path(args.output_csv)
        # si chemins relatifs, ils seront relatifs au répertoire courant
        if not in_path.is_absolute():
            in_path = Path.cwd() / in_path
        if not out_path.is_absolute():
            out_path = Path.cwd() / out_path
        predict_csv(pipe, in_path, args.text_col, out_path)
    else:
        demo(pipe)
