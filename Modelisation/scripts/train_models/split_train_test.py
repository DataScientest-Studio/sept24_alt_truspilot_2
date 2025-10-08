import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main(input_csv: str, output_dir: str, test_size: float = 0.2, random_state: int = 42) -> None:
    """
    Crée un découpage Train/Test stratifié pour un problème de classification à 5 classes (notes 1..5).

    Pourquoi ce script ?
      - Séparer un jeu de données en apprentissage (train) et évaluation (test) est indispensable
        pour mesurer une performance honnête (le modèle n'a jamais "vu" le test).
      - La "stratification" garde les mêmes proportions de classes dans train et test.

    Entrées possibles :
      1) Un CSV déjà nettoyé avec les colonnes: CleanText, Rating
      2) Un CSV brut avec : Title, Content, Rating (le script fusionne Title+Content)

    Sorties :
      - data/processed/train.csv (colonnes: CleanText, Rating)
      - data/processed/test.csv  (colonnes: CleanText, Rating)
    """

    # 1) Lecture du CSV d'entrée
    df = pd.read_csv(input_csv)

    # 2) Récupération du texte X et de la cible y selon les colonnes disponibles
    if {"CleanText", "Rating"}.issubset(df.columns):
        # Cas 1 : dataset déjà prétraité
        X = df["CleanText"].astype(str)
        y = df["Rating"].astype(int)
    elif {"Title", "Content", "Rating"}.issubset(df.columns):
        # Cas 2 : dataset brut → on construit FullText = Title + Content
        full = (
            df["Title"].fillna("").astype(str).str.strip() + " " +
            df["Content"].fillna("").astype(str).str.strip()
        ).str.strip()
        X = full
        y = pd.to_numeric(df["Rating"], errors="coerce").astype(int)
    else:
        # Message d'erreur clair pour les débutants
        raise ValueError(
            "Le CSV doit contenir soit (CleanText, Rating) soit (Title, Content, Rating).\n"
            "→ Astuce : exécuter d'abord preprocessing_advanced.py si besoin."
        )

    # 3) Découpage stratifié (préserve la proportion 1★..5★)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) Écriture des fichiers de sortie
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({"CleanText": X_train, "Rating": y_train}).to_csv(output / "train.csv", index=False)
    pd.DataFrame({"CleanText": X_test, "Rating": y_test}).to_csv(output / "test.csv", index=False)

    # 5) Petit récap lisible en console
    print(f"✔️ Fichiers écrits dans {output}: train.csv, test.csv")
    print(f"Tailles → train: {len(X_train)}, test: {len(X_test)}")
    print("Distribution train (proportions) :\n", y_train.value_counts(normalize=True).sort_index().round(3))
    print("Distribution test  (proportions) :\n", y_test.value_counts(normalize=True).sort_index().round(3))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split Train/Test stratifié pour avis Trustpilot (1..5)")
    parser.add_argument("--input", type=str, default="data/processed/trustpilot_dataset_final_cleaned.csv",
                        help="CSV d'entrée : soit CleanText+Rating, soit Title+Content+Rating")
    parser.add_argument("--outdir", type=str, default="data/processed",
                        help="Dossier de sortie pour train.csv et test.csv")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion du test (ex: 0.2 = 20%)")
    parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire pour reproductibilité")
    args = parser.parse_args()

    main(args.input, args.outdir, test_size=args.test_size, random_state=args.seed)
