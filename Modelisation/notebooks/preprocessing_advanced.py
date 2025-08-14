import pandas as pd
import re
import argparse
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from pathlib import Path

# Création auto des dossiers pipeline
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Télécharger les ressources NLTK si besoin
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def clean_text(text):
    # Minuscule
    text = text.lower()
    # Garde lettres + chiffres
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Supprime espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_stopwords(text, stop_words):
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text, lemmatizer):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def main(input_file, output_file):
    df = pd.read_csv(input_file)

    # Vérif colonnes
    if {"Title", "Content"}.issubset(df.columns):
        df["FullText"] = df["Title"].fillna("").astype(str).str.strip() + " " + df["Content"].fillna("").astype(str).str.strip()
    elif "FullText" in df.columns:
        pass  # déjà présent
    else:
        raise ValueError("Impossible de trouver 'FullText' OU 'Title'+'Content' dans le fichier d'entrée.")

    # Nettoyage regex
    df["CleanText"] = df["FullText"].apply(clean_text)

    # Stopwords anglais
    stop_words = set(stopwords.words('english'))
    df["CleanText"] = df["CleanText"].apply(lambda x: remove_stopwords(x, stop_words))

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    df["CleanText"] = df["CleanText"].apply(lambda x: lemmatize_text(x, lemmatizer))

    # Sauvegarde
    df_out = df[["CleanText", "Rating"]]
    df_out.to_csv(output_file, index=False, quoting=1)
    print(f"✅ Dataset prétraité exporté dans '{output_file}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement avancé du texte Trustpilot")
    parser.add_argument("--input", type=str, default="trustpilot_dataset_final_clean_regex.csv", help="Fichier CSV en entrée")
    parser.add_argument("--output", type=str, default="trustpilot_dataset_final_cleaned.csv", help="Fichier CSV en sortie")
    args = parser.parse_args()

    main(args.input, args.output)
