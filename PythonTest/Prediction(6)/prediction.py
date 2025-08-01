import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import os

# Fonctions pour charger les fichiers
def charger_csv():
    path = filedialog.askopenfilename(
        title="Sélectionner un fichier CSV à prédire",
        filetypes=[("Fichiers CSV", "*.csv")]
    )
    if path:
        csv_path.set(path)

def charger_modele():
    path = filedialog.askopenfilename(
        title="Sélectionner le fichier du modèle (.joblib)",
        filetypes=[("Fichiers joblib", "*.joblib")]
    )
    if path:
        model_path.set(path)

def charger_vectoriseur():
    path = filedialog.askopenfilename(
        title="Sélectionner le fichier du vectoriseur (.joblib)",
        filetypes=[("Fichiers joblib", "*.joblib")]
    )
    if path:
        vectorizer_path.set(path)

# Fonction principale de prédiction
def lancer_prediction():
    if not csv_path.get() or not model_path.get() or not vectorizer_path.get():
        messagebox.showerror("Erreur", "Veuillez sélectionner tous les fichiers.")
        return

    try:
        df = pd.read_csv(csv_path.get(), encoding="utf-8-sig", sep=';')
        if "Texte" not in df.columns:
            messagebox.showerror("Erreur", "Le fichier CSV doit contenir une colonne 'Texte'.")
            return
    except Exception as e:
        messagebox.showerror("Erreur lecture CSV", str(e))
        return

    try:
        clf = joblib.load(model_path.get())
        vectorizer = joblib.load(vectorizer_path.get())
    except Exception as e:
        messagebox.showerror("Erreur chargement modèle/vectoriseur", str(e))
        return

    try:
        X = df["Texte"].astype(str)
        X_vec = vectorizer.transform(X)
        predictions = clf.predict(X_vec)
        df["Prediction"] = predictions

        base, ext = os.path.splitext(csv_path.get())
        out_path = base + "_avec_predictions.csv"
        df.to_csv(out_path, index=False, encoding="utf-8-sig", sep=';', quoting=1) 

        messagebox.showinfo("Succès", f"Fichier sauvegardé avec les prédictions :\n{out_path}")
    except Exception as e:
        messagebox.showerror("Erreur prédiction", str(e))

# Interface graphique
root = tk.Tk()
root.title("Prédiction Sentiment")
root.geometry("600x350")

csv_path = tk.StringVar()
model_path = tk.StringVar()
vectorizer_path = tk.StringVar()

# Fichier CSV
tk.Label(root, text="1. Sélectionner un fichier CSV à prédire :").pack(pady=5)
tk.Button(root, text="📄 Choisir CSV", command=charger_csv).pack()
tk.Label(root, textvariable=csv_path, wraplength=550, fg="blue").pack()

# Modèle
tk.Label(root, text="2. Sélectionner le fichier du modèle :").pack(pady=5)
tk.Button(root, text="🤖 Choisir modèle", command=charger_modele).pack()
tk.Label(root, textvariable=model_path, wraplength=550, fg="blue").pack()

# Vectoriseur
tk.Label(root, text="3. Sélectionner le fichier du vectoriseur :").pack(pady=5)
tk.Button(root, text="🧠 Choisir vectoriseur", command=charger_vectoriseur).pack()
tk.Label(root, textvariable=vectorizer_path, wraplength=550, fg="blue").pack()

# Bouton lancer
tk.Button(root, text="🚀 Lancer la prédiction", command=lancer_prediction, font=("Arial", 12, "bold")).pack(pady=20)

root.mainloop()
