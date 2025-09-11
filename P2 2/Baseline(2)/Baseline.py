#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Auto-install deps if missing ---
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for p in ["pandas","sklearn","joblib","tk"]:
    try:
        ensure(p if p!="sklearn" else "scikit-learn")
    except Exception:
        pass  # tkinter peut être déjà présent selon l'OS

# --- Imports ---
import os
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class BaselineApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Baseline Sentiment - Count / TF-IDF")
        self.geometry("720x560")
        self.df = None
        self.csv_path = None

        root = ttk.Frame(self, padding=8)
        root.pack(fill="both", expand=True)
        # File row
        self.path_var = tk.StringVar(value="(aucun fichier)")
        ttk.Entry(root, textvariable=self.path_var, width=80).grid(row=0, column=0, sticky="we")
        ttk.Button(root, text="Ouvrir CSV", command=self.open_csv).grid(row=0, column=1, padx=6)
        # Columns row
        ttk.Label(root, text="Colonne texte:").grid(row=1, column=0, sticky="w", pady=(8,2))
        self.text_col = tk.StringVar(value="Texte")
        self.text_cb = ttk.Combobox(root, textvariable=self.text_col, state="readonly", width=40)
        self.text_cb.grid(row=1, column=1, sticky="w")
        ttk.Label(root, text="Colonne label:").grid(row=2, column=0, sticky="w")
        self.target_col = tk.StringVar(value="Sentiment")
        self.target_cb = ttk.Combobox(root, textvariable=self.target_col, state="readonly", width=40)
        self.target_cb.grid(row=2, column=1, sticky="w")

        # Jobs & params
        params = ttk.LabelFrame(root, text="Jobs & paramètres", padding=8)
        params.grid(row=3, column=0, columnspan=2, sticky="we", pady=8)
        self.job_count = tk.BooleanVar(value=True)
        self.job_tfidf = tk.BooleanVar(value=True)
        ttk.Checkbutton(params, text="CountVectorizer", variable=self.job_count).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(params, text="TF-IDF", variable=self.job_tfidf).grid(row=0, column=1, sticky="w")
        ttk.Label(params, text="test_size (0-1):").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.test_size = tk.StringVar(value="0.2")
        ttk.Entry(params, textvariable=self.test_size, width=8).grid(row=1, column=1, sticky="w", pady=(6,0))
        ttk.Label(params, text="Dossier modèles:").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.outdir = tk.StringVar(value="modeles")
        ttk.Entry(params, textvariable=self.outdir, width=24).grid(row=2, column=1, sticky="w", pady=(6,0))

        # Actions
        ttk.Button(root, text="Entraîner", command=self.train).grid(row=4, column=0, columnspan=2, pady=8)

        # Output
        self.out = tk.Text(root, height=18)
        self.out.grid(row=5, column=0, columnspan=2, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(5, weight=1)

    def log(self, s):
        self.out.insert("end", s + "\n")
        self.out.see("end")

    def open_csv(self):
        p = filedialog.askopenfilename(title="Choisir un CSV", filetypes=[("CSV","*.csv"),("Tous","*.*")])
        if not p: return
        try:
            try:
                df = pd.read_csv(p, sep=None, engine="python", dtype=str, encoding="utf-8")
            except Exception:
                df = None
                for sep in [";","\t",","]:
                    try:
                        df = pd.read_csv(p, sep=sep, dtype=str, encoding="utf-8")
                        break
                    except Exception:
                        pass
                if df is None: raise
            self.df = df
            self.csv_path = p
            self.path_var.set(p)
            cols = list(df.columns)
            self.text_cb["values"] = cols
            self.target_cb["values"] = cols
            # auto-guess
            for c in ["Texte","CleanText","text","Text"]:
                if c in cols: self.text_col.set(c); break
            for c in ["Sentiment","Rating","label","target"]:
                if c in cols: self.target_col.set(c); break
            self.log(f"✅ CSV chargé: {len(df)} lignes, colonnes={cols}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Lecture CSV: {e}")

    def _train_job(self, df, text_col, target_col, vec_kind, test_size, outdir):
        X = df[text_col].fillna("").astype(str)
        y = df[target_col].astype(str)
        try:
            ts = float(test_size)
            if not (0 < ts < 1): raise ValueError
        except Exception:
            raise ValueError("test_size doit être un flottant entre 0 et 1 (ex: 0.2)")

        X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=ts, random_state=42, stratify=y)

        if vec_kind == "count":
            vectorizer = CountVectorizer(lowercase=True, ngram_range=(1,2), min_df=2)
            tag = "count"
        else:
            vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1,2), min_df=2)
            tag = "tfidf"

        Xtr = vectorizer.fit_transform(X_tr)
        Xva = vectorizer.transform(X_va)

        clf = LogisticRegression(max_iter=500, class_weight="balanced")
        clf.fit(Xtr, y_tr)

        y_pred = clf.predict(Xva)
        acc = accuracy_score(y_va, y_pred)
        f1m = f1_score(y_va, y_pred, average="macro")
        f1w = f1_score(y_va, y_pred, average="weighted")

        self.log(f"[{tag}] Accuracy={acc:.4f} | F1-macro={f1m:.4f} | F1-weighted={f1w:.4f}")
        self.log(classification_report(y_va, y_pred, zero_division=0))

        os.makedirs(outdir, exist_ok=True)
        model_path = os.path.join(outdir, f"modele_sentiment_{tag}.joblib")
        vect_path  = os.path.join(outdir, f"vectorizer_{tag}.joblib")
        dump(clf, model_path)
        dump(vectorizer, vect_path)
        self.log(f"✔ Modèle: {model_path}")
        self.log(f"✔ Vectorizer: {vect_path}")

    def train(self):
        if self.df is None:
            messagebox.showinfo("Info", "Charge d'abord un CSV.")
            return
        text_col = self.text_col.get()
        target_col = self.target_col.get()
        if text_col not in self.df.columns or target_col not in self.df.columns:
            messagebox.showerror("Erreur", "Colonnes texte/label invalides.")
            return
        selected = []
        if self.job_count.get(): selected.append("count")
        if self.job_tfidf.get(): selected.append("tfidf")
        if not selected:
            messagebox.showinfo("Info", "Sélectionne au moins un job (Count / TF-IDF).")
            return
        self.out.delete("1.0","end")
        self.log(f"Fichier: {self.csv_path}")
        self.log(f"Texte='{text_col}'  Label='{target_col}'  Jobs={selected}\n")
        for j in selected:
            try:
                self._train_job(self.df, text_col, target_col, j, self.test_size.get(), self.outdir.get())
            except Exception as e:
                messagebox.showerror("Erreur", f"Echec job '{j}': {e}")
                return
        self.log("✅ Terminé.")

if __name__ == "__main__":
    BaselineApp().mainloop()
