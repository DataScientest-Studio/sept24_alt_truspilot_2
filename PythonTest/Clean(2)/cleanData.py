import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import re
import string
import unicodedata
import os
import csv

# Tentative d'import de NLTK pour les stopwords
try:
    import nltk
    from nltk.corpus import stopwords
    nltk.download('stopwords', quiet=True)
    STOPWORDS = set(stopwords.words('french'))
except:
    STOPWORDS = set()
    print("⚠️ NLTK non disponible, les stopwords ne seront pas supprimés.")

def remove_accents(text):
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = remove_accents(text)
    text = re.sub(rf"[{re.escape(string.punctuation)}\d]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if STOPWORDS:
        text = " ".join(w for w in text.split() if w not in STOPWORDS)
    return text

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

def process_file(file_path):
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8-sig')
        if "Texte" not in df.columns or "Note" not in df.columns:
            messagebox.showerror("Erreur", "Le fichier doit contenir les colonnes 'Texte' et 'Note'.")
            return

        df["Texte"] = df["Texte"].apply(clean_text)
        df["Sentiment"] = df["Note"].apply(compute_sentiment)

        # Conserver les colonnes utiles
        keep_cols = ["Auteur", "Date", "Note", "Texte", "Sentiment"]
        df = df[[c for c in keep_cols if c in df.columns]]

        # Export du fichier
        base, _ = os.path.splitext(file_path)
        output_file = base + "_nettoye.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig', sep=';', quoting=csv.QUOTE_ALL)

        messagebox.showinfo("Succès", f"Fichier nettoyé et étiqueté enregistré :\n{output_file}")
    except Exception as e:
        messagebox.showerror("Erreur", str(e))

def open_file():
    fpath = filedialog.askopenfilename(
        title="Sélectionner un fichier CSV",
        filetypes=[("Fichiers CSV", "*.csv")]
    )
    if fpath:
        process_file(fpath)

# Interface Tkinter
root = tk.Tk()
root.title("Nettoyage + Étiquetage Sentiment")
root.geometry("420x150")
tk.Label(root, text="Sélectionnez un fichier CSV contenant les colonnes 'Texte' et 'Note'").pack(pady=20)
tk.Button(root, text="Choisir un fichier CSV", command=open_file).pack(pady=10)
root.mainloop()
