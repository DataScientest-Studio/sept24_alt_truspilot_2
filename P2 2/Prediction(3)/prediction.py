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
        pass

# --- Imports ---
import pandas as pd
import joblib
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.pipeline import Pipeline

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(c):
        if isinstance(c, str):
            c = c.replace("\\ufeff","").replace("\\uFEFF","").strip().strip("'").strip('"')
        return c
    df = df.copy()
    df.columns = [clean(c) for c in df.columns]
    return df

class PredictApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Prédiction (Pipeline ou Classif + Vectorizer)")
        self.geometry("820x580")
        self.df = None
        self.model = None
        self.vectorizer = None
        self.model_path = None
        self.vec_path = None
        self.csv_path = None

        root = ttk.Frame(self, padding=8); root.pack(fill="both", expand=True)

        # Row 0: model
        ttk.Label(root, text="Modèle (.joblib) :").grid(row=0, column=0, sticky="w")
        self.model_var = tk.StringVar(value="(aucun)")
        ttk.Entry(root, textvariable=self.model_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Choisir...", command=self.pick_model).grid(row=0, column=2, padx=6)

        # Row 1: vectorizer (optional)
        ttk.Label(root, text="Vectorizer (.joblib, optionnel) :").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.vec_var = tk.StringVar(value="(non requis si modèle=Pipeline)")
        ttk.Entry(root, textvariable=self.vec_var, width=70).grid(row=1, column=1, sticky="we", pady=(6,0))
        ttk.Button(root, text="Choisir...", command=self.pick_vectorizer).grid(row=1, column=2, padx=6, pady=(6,0))

        # Row 2: csv
        ttk.Label(root, text="CSV d'entrée :").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.csv_var = tk.StringVar(value="(aucun)")
        ttk.Entry(root, textvariable=self.csv_var, width=70).grid(row=2, column=1, sticky="we", pady=(6,0))
        ttk.Button(root, text="Ouvrir...", command=self.pick_csv).grid(row=2, column=2, padx=6, pady=(6,0))

        # Row 3-4: columns
        ttk.Label(root, text="Colonne texte (obligatoire) :").grid(row=3, column=0, sticky="w", pady=(8,0))
        self.text_col = tk.StringVar(value="CleanText")
        self.text_cb = ttk.Combobox(root, textvariable=self.text_col, state="readonly", width=40)
        self.text_cb.grid(row=3, column=1, sticky="w", pady=(8,0))

        ttk.Label(root, text="Colonne cible y_true (facultatif) :").grid(row=4, column=0, sticky="w", pady=(6,0))
        self.target_col = tk.StringVar(value="")
        self.target_cb = ttk.Combobox(root, textvariable=self.target_col, state="readonly", width=40)
        self.target_cb.grid(row=4, column=1, sticky="w", pady=(6,0))

        # Actions
        ttk.Button(root, text="Prédire et sauvegarder...", command=self.predict_and_save).grid(row=5, column=0, columnspan=3, pady=10)

        # Output
        self.out = tk.Text(root, height=18)
        self.out.grid(row=6, column=0, columnspan=3, sticky="nsew")
        root.columnconfigure(1, weight=1)
        root.rowconfigure(6, weight=1)

    def log(self, s):
        self.out.insert("end", s+"\n")
        self.out.see("end")

    def pick_model(self):
        p = filedialog.askopenfilename(title="Choisir un modèle .joblib", filetypes=[("joblib","*.joblib"),("Tous","*.*")])
        if not p: return
        try:
            self.model = joblib.load(p)
            self.model_path = p
            self.model_var.set(p)
            kind = "Pipeline" if isinstance(self.model, Pipeline) else type(self.model).__name__
            self.log(f"✅ Modèle chargé: {p}  (type: {kind})")
            if isinstance(self.model, Pipeline):
                self.vec_var.set("(non requis, modèle=Pipeline)")
                self.vectorizer = None
                self.vec_path = None
        except Exception as e:
            messagebox.showerror("Erreur", f"Chargement modèle:\n{e}")

    def pick_vectorizer(self):
        p = filedialog.askopenfilename(title="Choisir un vectorizer .joblib", filetypes=[("joblib","*.joblib"),("Tous","*.*")])
        if not p: return
        try:
            self.vectorizer = joblib.load(p)
            self.vec_path = p
            self.vec_var.set(p)
            self.log(f"✅ Vectorizer chargé: {p}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Chargement vectorizer:\n{e}")

    def pick_csv(self):
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
                if df is None:
                    raise
            df = sanitize_columns(df)
            self.df = df
            self.csv_path = p
            self.csv_var.set(p)
            cols = list(df.columns)
            self.text_cb["values"] = cols
            self.target_cb["values"] = [""] + cols
            # auto-guess common columns
            for c in ["CleanText","Texte","text","Text","content","Content"]:
                if c in cols:
                    self.text_col.set(c); break
            for c in ["Sentiment","Rating","label","target","y_true"]:
                if c in cols:
                    self.target_col.set(c); break
            self.log(f"✅ CSV chargé: {len(df)} lignes, colonnes={cols}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Lecture CSV:\n{e}")

    def predict_and_save(self):
        if self.model is None:
            messagebox.showinfo("Info", "Sélectionne d'abord un modèle .joblib.")
            return
        if self.df is None:
            messagebox.showinfo("Info", "Sélectionne un CSV d'entrée.")
            return
        text_col = self.text_col.get()
        if text_col not in self.df.columns:
            messagebox.showerror("Erreur", f"Colonne texte '{text_col}' introuvable.")
            return

        X_text = self.df[text_col].fillna("").astype(str)

        try:
            if isinstance(self.model, Pipeline):
                # Directly predict from raw text
                y_pred = self.model.predict(X_text)
            else:
                # Need a separate vectorizer
                if self.vectorizer is None:
                    messagebox.showwarning("Vectorizer manquant",
                        "Le modèle n'est pas un Pipeline. Sélectionne un vectorizer .joblib correspondant (Count/Tfidf).")
                    return
                X_vec = self.vectorizer.transform(X_text)
                y_pred = self.model.predict(X_vec)
        except Exception as e:
            messagebox.showerror("Erreur", f"Échec prédiction:\n{e}")
            return

        out_cols = {}
        tgt = self.target_col.get().strip()
        if tgt and tgt in self.df.columns:
            out_cols["y_true"] = self.df[tgt].astype(str).values
        out_cols["y_pred"] = y_pred
        out_df = pd.DataFrame(out_cols)

        save = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="predictions.csv",
                                            filetypes=[("CSV","*.csv")])
        if not save: return
        try:
            out_df.to_csv(save, index=False, encoding="utf-8-sig", sep=";")
            self.log(f"✅ Prédictions sauvegardées: {save}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Sauvegarde échouée:\n{e}")

if __name__ == "__main__":
    PredictApp().mainloop()
