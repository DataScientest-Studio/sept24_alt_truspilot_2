import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import string
import unicodedata
import os
import csv
import sys

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

# Vérification et installation automatique de spaCy et du modèle français
try:
    import spacy
except ImportError:
    import subprocess
    result = messagebox.askyesno(
        "Module manquant",
        "Le module spaCy n'est pas installé.\nSouhaitez-vous l'installer maintenant ?"
    )
    if result:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "spacy"])
        import spacy
    else:
        sys.exit(1)


# Tentative de chargement du modèle
try:
    nlp = spacy.load('fr_core_news_sm')
except OSError:
    try:
        # Installation automatique du modèle
        from spacy.cli import download
        download('fr_core_news_sm')
        nlp = spacy.load('fr_core_news_sm')
    except Exception as e:
        messagebox.showerror(
            "Modèle manquant",
            "Le modèle fr_core_news_sm n'a pas pu être téléchargé automatiquement.\n"
            f"Erreur : {e}\n"
            "Veuillez exécuter : python -m spacy download fr_core_news_sm"
        )
        sys.exit(1)

# Télécharger les ressources NLTK si nécessaire
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Initialisation des outils
STOPWORDS = set(stopwords.words('french'))
STEMMER   = FrenchStemmer()

# Fallback pour la lemmatisation anglaise (rarement utilisé)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def remove_accents(text: str) -> str:
    """Supprime les accents d’une chaîne."""
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def preprocess_text(text: str) -> str:
    """Nettoie, tokenise, applique stemming + lemmatisation, et supprime les doublons."""
    if pd.isna(text):
        return ""
    # minuscules & suppression des accents
    text = remove_accents(text.lower())
    # retirer ponctuation et chiffres
    text = re.sub(f"[{re.escape(string.punctuation)}\\d]", " ", text)
    # espaces multiples → simple
    text = re.sub(r"\s+", " ", text).strip()

    # tokens de ≥2 lettres
    tokens = re.findall(r"\b[a-zàâäéèêëîïôöùûüÿçœæ]{2,}\b", text)
    # filtrage stopwords
    tokens = [t for t in tokens if t not in STOPWORDS]

    processed = []
    seen = set()
    # spaCy pour lemmatisation en français
    doc = nlp(" ".join(tokens))
    for token in doc:
        if token.is_alpha and len(token) > 1:
            lemma = token.lemma_.lower()
            if lemma not in STOPWORDS:
                stem = STEMMER.stem(lemma)
                if stem not in seen:
                    seen.add(stem)
                    processed.append(stem)

    return ' '.join(processed)


def compute_sentiment(note):
    try:
        note = float(note)
        if note >= 4:
            return 1
        elif note <= 2:
            return 0
        else:
            return None  # Neutre
    except:
        return None

def process_file(file_path: str):
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', encoding='utf-8-sig')
    except Exception as e:
        messagebox.showerror(
            "Erreur de lecture",
            f"Impossible de charger le fichier :\n{e}"
        )
        return

    if 'Texte' not in df.columns:
        messagebox.showerror("Erreur", "La colonne 'Texte' est introuvable.")
        return

    df['Texte'] = df['Texte'].apply(preprocess_text)
    df["Sentiment"] = df["Note"].apply(compute_sentiment)
    wanted = ['Auteur', 'Date', 'Note','Sentiment', 'Texte']
    cols = [c for c in wanted if c in df.columns]
    df = df[cols]

    base, _ = os.path.splitext(file_path)
    out_path = f"{base}_preprocessed.csv"
    try:
        df.to_csv(
            out_path,
            index=False,
            encoding='utf-8-sig',
            sep=';',
            quoting=csv.QUOTE_ALL
        )
        messagebox.showinfo(
            "Terminé",
            f"Fichier prétraité enregistré sous :\n{out_path}"
        )
    except Exception as e:
        messagebox.showerror("Erreur d'écriture", str(e))

def open_file():
    path = filedialog.askopenfilename(
        title="Sélectionnez un fichier CSV",
        filetypes=[("Fichiers CSV", "*.csv")]
    )
    if path:
        process_file(path)

if __name__ == '__main__':
    root = tk.Tk()
    root.title("Préprocesseur Avancé")
    root.geometry("500x160")
    tk.Label(root, text="Sélectionnez un fichier CSV à prétraiter").pack(pady=20)
    tk.Button(root, text="Choisir le fichier...", command=open_file).pack(pady=10)
    root.mainloop()
