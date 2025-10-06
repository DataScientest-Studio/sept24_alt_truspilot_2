import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path


def main(input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):
    df = pd.read_csv(input_csv)

    # Accepte soit (CleanText, Rating) déjà nettoyé, soit (Title, Content, Rating)
    if {"CleanText", "Rating"}.issubset(df.columns):
        X = df["CleanText"].astype(str)
        y = df["Rating"].astype(int)
    elif {"Title", "Content", "Rating"}.issubset(df.columns):
        full = (
            df["Title"].fillna("").astype(str).str.strip() + " " +
            df["Content"].fillna("").astype(str).str.strip()
        ).str.strip()
        X = full
        y = pd.to_numeric(df["Rating"], errors="coerce").astype(int)
    else:
        raise ValueError("Le CSV doit contenir soit (CleanText, Rating) soit (Title, Content, Rating).")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"CleanText": X_train, "Rating": y_train}).to_csv(output / "train.csv", index=False)
    pd.DataFrame({"CleanText": X_test, "Rating": y_test}).to_csv(output / "test.csv", index=False)
    print(f"✔️ Fichiers écrits dans {output}: train.csv, test.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/trustpilot_dataset_final_cleaned.csv",
                        help="CSV d'entrée (CleanText, Rating OU Title, Content, Rating)")
    parser.add_argument("--outdir", type=str, default="data/processed",
                        help="Dossier de sortie pour train.csv et test.csv")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args.input, args.outdir, test_size=args.test_size, random_state=args.seed)
