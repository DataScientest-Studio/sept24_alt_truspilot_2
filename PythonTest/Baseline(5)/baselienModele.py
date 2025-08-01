import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import joblib
import os
import subprocess
import sys

# Installation automatique des packages si manquants
def install_if_missing(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for pkg in ['pandas', 'sklearn', 'joblib']:
    install_if_missing(pkg)

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Fonction principale
def lancer_analyse():
    modele = vect_choice.get()
    if not filepath.get():
        messagebox.showerror("Erreur", "Veuillez choisir un fichier CSV.")
        return

    try:
        df = pd.read_csv(filepath.get(), encoding="utf-8-sig", sep=';')
        df.columns = df.columns.str.strip().str.replace('"', '').str.replace("'", "")
    except Exception as e:
        messagebox.showerror("Erreur lecture CSV", str(e))
        return

    if "Texte" not in df.columns or "Sentiment" not in df.columns:
        messagebox.showerror("Erreur", "Le fichier doit contenir les colonnes 'Texte' et 'Sentiment'")
        return

    df["Sentiment"] = pd.to_numeric(df["Sentiment"], errors="coerce")
    df = df.dropna(subset=["Texte", "Sentiment"])
    df["Sentiment"] = df["Sentiment"].astype(int)

    X, y = df["Texte"], df["Sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

    # Vectorisation avec n-grammes
    if modele == "CountVectorizer":
        vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=2)
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2)

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modèle : régression logistique équilibrée
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=0)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)

    # Sauvegarde modèle et vectoriseur
    output_dir = "modeles"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "count" if modele == "CountVectorizer" else "tfidf"
    model_path = os.path.join(output_dir, f"modele_sentiment_{suffix}.joblib")
    vectorizer_path = os.path.join(output_dir, f"vectorizer_{suffix}.joblib")
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    messagebox.showinfo("Modèle enregistré", f"Modèle et vectoriseur enregistrés dans :\n{output_dir}")


    # Affichage des résultats
    rapport = classification_report(y_test, y_pred, output_dict=True)
    show_report_popup(rapport, y_test, y_pred)

    # Export CSV avec comparaison
    df_result = pd.DataFrame({
        "Texte": X_test,
        "Vrai": y_test,
        "Prévu": y_pred
    })
    df_result["Correct"] = df_result["Vrai"] == df_result["Prévu"]
    df_result.to_csv("resultats_test_predictions.csv", sep=";", encoding="utf-8-sig", index=False)

# Affichage popup des métriques
def show_report_popup(report, y_test, y_pred):
    from sklearn.metrics import accuracy_score

    metrique = report['weighted avg']
    accuracy = accuracy_score(y_test, y_pred)

    result = (
        f"Accuracy : {accuracy:.2f}\n"
        f"Precision : {metrique['precision']:.2f}\n"
        f"Recall    : {metrique['recall']:.2f}\n"
        f"F1-score  : {metrique['f1-score']:.2f}"
    )

    popup = tk.Toplevel(root)
    popup.title("Résultats du modèle")
    popup.geometry("300x150")
    tk.Label(popup, text=result, justify="left", font=("Arial", 12)).pack(pady=20)

# Interface utilisateur
root = tk.Tk()
root.title("Analyse de Sentiment - Modèle Amélioré")
root.geometry("500x300")

filepath = tk.StringVar()
vect_choice = tk.StringVar(value="CountVectorizer")

tk.Label(root, text="1. Choisir un fichier CSV prétraité :").pack(pady=5)
tk.Button(root, text="Parcourir...", command=lambda: filepath.set(filedialog.askopenfilename())).pack()
tk.Label(root, textvariable=filepath, wraplength=400).pack(pady=5)

tk.Label(root, text="2. Choisir le type de vectorisation :").pack(pady=10)
tk.Radiobutton(root, text="CountVectorizer", variable=vect_choice, value="CountVectorizer").pack()
tk.Radiobutton(root, text="TF-IDF Vectorizer", variable=vect_choice, value="TfidfVectorizer").pack()

tk.Button(root, text="🚀 Lancer l'analyse", command=lancer_analyse, font=("Arial", 12)).pack(pady=20)

root.mainloop()
