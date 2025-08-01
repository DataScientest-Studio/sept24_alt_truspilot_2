import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd

import matplotlib.pyplot as plt

import subprocess
import sys



try:
    from wordcloud import WordCloud
except ImportError:
    print("WordCloud non installé. Installation en cours...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
    from wordcloud import WordCloud

def install_if_missing(package, module=None):
    module = module or package
    try:
        __import__(module)
    except ImportError:
        print(f"{package} manquant. Installation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Liste des dépendances à vérifier
install_if_missing("wordcloud")
install_if_missing("pandas")
install_if_missing("matplotlib")
install_if_missing("nltk")

from nltk.corpus import stopwords
import nltk

# Télécharger les stopwords français si nécessaire
try:
    stop_words = set(stopwords.words('french'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('french'))


def generer_nuage_mots():
    # Sélection du fichier CSV
    chemin_fichier = filedialog.askopenfilename(
        title="Sélectionner un fichier CSV",
        filetypes=[("Fichier CSV", "*.csv")]
    )
    
    if not chemin_fichier:
        return  # L'utilisateur a annulé

    # Lecture du fichier CSV
    try:
        df = df = pd.read_csv(chemin_fichier, encoding="utf-8-sig", sep=';')
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")

    except Exception as e:
        messagebox.showerror("Erreur de lecture", f"Impossible de lire le fichier :\n{e}")
        return

    print("Colonnes du fichier :", list(df.columns))

    # Vérification de la colonne
    if "Texte" not in df.columns:
        messagebox.showerror("Erreur", "La colonne 'Texte_nettoye' est absente du fichier CSV.")
        return

    # Fusion de tous les textes
    texte_total = " ".join(str(t) for t in df["Texte"].dropna())

    # Création du nuage de mots
    nuage = WordCloud(
        background_color="white",
        max_words=150,
        stopwords=stop_words,
        max_font_size=60,
        width=800,
        height=400,
        random_state=42
    )

    # Affichage
    plt.figure(figsize=(10, 5))
    nuage.generate(texte_total)
    plt.imshow(nuage, interpolation='bilinear')
    plt.axis("off")
    plt.title("Nuage de mots - Texte nettoyé", fontsize=14)
    plt.show()

# Interface graphique
fenetre = tk.Tk()
fenetre.title("Générateur de nuage de mots")
fenetre.geometry("420x200")

# Widgets
titre = tk.Label(fenetre, text="Sélectionnez un fichier CSV contenant la colonne 'Texte_nettoye'", wraplength=380, justify="center")
titre.pack(pady=20)

bouton = tk.Button(fenetre, text="Choisir un fichier...", command=generer_nuage_mots, font=("Arial", 12))
bouton.pack(pady=10)

fenetre.mainloop()
