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
    """
    Nettoyage simple :
    - passe en minuscules
    - conserve lettres/chiffres uniquement
    - retire ponctuation, symboles, espaces multiples
    """
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # garde uniquement a-z et chiffres
    text = re.sub(r"\s+", " ", text).strip()   # retire espaces multiples
    return text

def remove_stopwords(text, stop_words):
    """Supprime les mots très fréquents (stopwords) des langues sélectionnées."""
    return " ".join([word for word in text.split() if word not in stop_words])

def lemmatize_text(text, lemmatizer):
    """Réduit les mots à leur forme canonique (ex: running → run)."""
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def main(input_file, output_file):
    # Lecture du CSV brut ou regex-cleané
    df = pd.read_csv(input_file)

    # Vérifie la présence du texte
    if {"Title", "Content"}.issubset(df.columns):
        df["FullText"] = (
            df["Title"].fillna("").astype(str).str.strip() + " " +
            df["Content"].fillna("").astype(str).str.strip()
        )
    elif "FullText" in df.columns:
        pass  # déjà présent
    else:
        raise ValueError("Le fichier doit contenir 'FullText' OU 'Title'+'Content'.")

    # 1️⃣ Nettoyage basique (regex)
    df["CleanText"] = df["FullText"].apply(clean_text)

    # 2️⃣ Stopwords multi-langues (anglais + français)
    stop_words = set(stopwords.words('english'))
    try:
        stop_words_fr = set(stopwords.words('french'))
        stop_words.update(stop_words_fr)
        print(f"🗑️ Stopwords utilisés : {len(stop_words)} (EN + FR)")
    except LookupError:
        print("⚠️ Stopwords français non trouvés. Téléchargement : nltk.download('stopwords')")
        nltk.download('stopwords')
        stop_words.update(set(stopwords.words('french')))

    df["CleanText"] = df["CleanText"].apply(lambda x: remove_stopwords(x, stop_words))

    # 3️⃣ Lemmatisation
    lemmatizer = WordNetLemmatizer()
    df["CleanText"] = df["CleanText"].apply(lambda x: lemmatize_text(x, lemmatizer))

    # 4️⃣ Sauvegarde (CleanText + Rating)
    df_out = df[["CleanText", "Rating"]]
    df_out.to_csv(output_file, index=False, quoting=1)
    print(f"✅ Dataset prétraité exporté dans '{output_file}' ({len(df_out)} lignes).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement avancé (EN + FR) du texte Trustpilot")
    parser.add_argument("--input", type=str, default="trustpilot_dataset_final_clean_regex.csv",
                        help="Fichier CSV en entrée (avec Title/Content/FullText)")
    parser.add_argument("--output", type=str, default="trustpilot_dataset_final_cleaned.csv",
                        help="Fichier CSV en sortie (CleanText, Rating)")
    args = parser.parse_args()

    main(args.input, args.output)
