# notebooks/preprocessing_full.py
# Usage :
#   python notebooks/preprocessing_full.py
#   (ou avec chemins personnalisés)
#   python notebooks/preprocessing_full.py --input "notebooks/trustpilot_dataset_final.csv" --output "notebooks/trustpilot_dataset_final_clean_regex.csv"

import argparse
import pandas as pd
import re
from pathlib import Path

# Création auto des dossiers pipeline
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# --- Patterns regex (cf. cours) ---
URL_PATTERN = re.compile(r"https?://[A-Za-z0-9\.\-/_%#\?\=&]+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<.*?>")  # lazy
DIGITS_PATTERN = re.compile(r"\d+")
PUNCT_PATTERN = re.compile(r"[^\w\s]", flags=re.UNICODE)  # ponctuation/symboles
SEE_MORE_PATTERN = re.compile(r"\bsee\s+more\b", re.IGNORECASE)
ELLIPSIS_PATTERN = re.compile(r"\.{2,}")  # … / ...
UNDERSCORE_PATTERN = re.compile(r"_+")
MULTISPACE_PATTERN = re.compile(r"\s+")
TRIM_EDGES = re.compile(r"^\s+|\s+$")

# --- Fonctions ---
def regex_clean(text: str) -> str:
    """Nettoyage STRICTEMENT via expressions régulières (module Text Mining - Regex)."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    t = text

    # Suppressions directes
    t = HTML_TAG_PATTERN.sub(" ", t)
    t = URL_PATTERN.sub(" ", t)
    t = EMAIL_PATTERN.sub(" ", t)
    t = SEE_MORE_PATTERN.sub(" ", t)

    # Normalisations légères
    t = ELLIPSIS_PATTERN.sub(" ", t)
    t = UNDERSCORE_PATTERN.sub(" ", t)

    # Nettoyage caractères
    ##t = DIGITS_PATTERN.sub(" ", t)  # chiffres → espace (à commenter si on veut les garder)
    t = PUNCT_PATTERN.sub(" ", t)   # ponctuation/symboles → espace

    # Espaces
    t = MULTISPACE_PATTERN.sub(" ", t)
    t = TRIM_EDGES.sub("", t)

    return t

def regex_counts(text: str) -> dict:
    """Petites features dérivées par regex."""
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)
    return {
        "url_count": len(URL_PATTERN.findall(text)),
        "email_count": len(EMAIL_PATTERN.findall(text)),
        "digit_count": sum(len(m) for m in DIGITS_PATTERN.findall(text)),
        "punct_count": len(PUNCT_PATTERN.findall(text)),
    }

def main(input_path: str, output_path: str):
    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"[i] Lecture : {input_path}")
    df = pd.read_csv(input_path)

    print(f"[i] Colonnes détectées : {list(df.columns)}")

    # === 1. Nettoyage structure ===
    # Supprimer lignes sans texte (Title & Content vides)
    if "Content" in df.columns:
        df = df.dropna(subset=["Content", "Rating"])
    else:
        raise ValueError("Le fichier doit contenir au minimum 'Content' et 'Rating'.")

    # Convertir Rating en numérique et filtrer 1–5
    df["Rating"] = pd.to_numeric(df["Rating"], errors="coerce")
    df = df.dropna(subset=["Rating"])
    df = df[df["Rating"].isin([1, 2, 3, 4, 5])]

    # === 2. Fusion Title+Content ===
    if "Title" not in df.columns:
        df["Title"] = ""  # si pas de titre dans le fichier
    df["FullText"] = (
        df["Title"].fillna("").astype(str).str.strip() + " " +
        df["Content"].fillna("").astype(str).str.strip()
    ).str.strip()

    # === 3. Compteurs regex (avant nettoyage) ===
    counts = df["FullText"].apply(regex_counts).apply(pd.Series)
    df = pd.concat([df, counts], axis=1)

    # === 4. Nettoyage texte (regex avancé) ===
    df["CleanText_regex"] = df["FullText"].apply(regex_clean)

    # === 5. Export ===
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Fichier écrit : {output_path}")
    print(df[["CleanText_regex", "Rating"]].head(5))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="trustpilot_dataset_final.csv")
    parser.add_argument("--output", type=str, default="trustpilot_dataset_final_clean_regex.csv")
    args = parser.parse_args()
    main(args.input, args.output)
