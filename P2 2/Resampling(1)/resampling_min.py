#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Auto-install libs if missing ---
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for p in ["pandas","sklearn","imblearn","tk"]:
    try:
        ensure(p if p!="sklearn" else "scikit-learn")
    except Exception:
        pass  # tkinter peut déjà être présent selon la plateforme

# --- Imports ---
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import Counter

# imblearn (si dispo)
try:
    from imblearn.over_sampling import RandomOverSampler, SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    IMB = True
except Exception:
    IMB = False

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Resampling CSV")
        self.geometry("560x420")
        self.df = None
        # UI minimale
        frm = ttk.Frame(self, padding=8); frm.pack(fill="both", expand=True)
        self.path = tk.StringVar(value="(aucun)")
        ttk.Entry(frm, textvariable=self.path, width=60).grid(row=0, column=0, sticky="we")
        ttk.Button(frm, text="Ouvrir CSV", command=self.open_csv).grid(row=0, column=1, padx=6)
        ttk.Label(frm, text="Colonne cible:").grid(row=1, column=0, sticky="w", pady=6)
        self.yvar = tk.StringVar(value="Rating")
        self.yentry = ttk.Entry(frm, textvariable=self.yvar, width=20)
        self.yentry.grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="Méthode:").grid(row=2, column=0, sticky="w")
        self.meth = tk.StringVar(value="Aucun")
        cb = ttk.Combobox(frm, state="readonly", textvariable=self.meth,
                          values=["Aucun","RandomOverSampler","RandomUnderSampler","SMOTE"], width=24)
        cb.grid(row=2, column=1, sticky="w")
        ttk.Button(frm, text="Resampler et sauvegarder", command=self.run).grid(row=3, column=0, columnspan=2, pady=10)
        self.out = tk.Text(frm, height=12); self.out.grid(row=4, column=0, columnspan=2, sticky="nsew")
        frm.columnconfigure(0, weight=1); frm.rowconfigure(4, weight=1)

    def log(self, s): self.out.insert("end", s+"\n"); self.out.see("end")

    def open_csv(self):
        p = filedialog.askopenfilename(title="Choisir un CSV", filetypes=[("CSV","*.csv"),("Tous","*.*")])
        if not p: return
        self.path.set(p)
        try:
            import pandas as pd
            try:
                df = pd.read_csv(p, sep=None, engine="python", dtype=str, encoding="utf-8")
            except Exception:
                df = None
                for sep in [";","\t",","]:
                    try:
                        df = pd.read_csv(p, sep=sep, dtype=str, encoding="utf-8")
                        break
                    except Exception: pass
                if df is None: raise
            self.df = df
            self.log(f"Chargé: {len(df)} lignes, colonnes={list(df.columns)}")
        except Exception as e:
            messagebox.showerror("Erreur", str(e))

    def _internal_over(self, X, y):
        counts = Counter(y)
        maxn = max(counts.values())
        parts = []
        for cls, n in counts.items():
            idx = y[y==cls].index
            if n < maxn:
                add = idx.to_series().sample(maxn-n, replace=True, random_state=42).values
                idx = idx.append(add)
            parts.append(idx)
        new_idx = parts[0].union_many(parts[1:])
        return X.loc[new_idx], y.loc[new_idx]

    def _internal_under(self, X, y):
        counts = Counter(y)
        minn = min(counts.values())
        parts = []
        for cls, n in counts.items():
            idx = y[y==cls].index
            if n > minn:
                idx = idx.to_series().sample(minn, replace=False, random_state=42).values
            parts.append(idx)
        new_idx = parts[0].union_many(parts[1:])
        return X.loc[new_idx], y.loc[new_idx]

    def run(self):
        if self.df is None: 
            messagebox.showinfo("Info","Charge un CSV."); return
        ycol = self.yvar.get()
        if ycol not in self.df.columns:
            messagebox.showerror("Erreur", f"Colonne '{ycol}' absente."); return
        method = self.meth.get()
        y = self.df[ycol]
        X = self.df.drop(columns=[ycol])

        self.out.delete("1.0","end")
        self.log("Avant: " + str(Counter(y)))
        if method == "Aucun":
            res = self.df.copy()
        elif method == "RandomOverSampler":
            if IMB:
                ros = RandomOverSampler(random_state=42)
                Xr, yr = ros.fit_resample(X, y)
                res = pd.concat([Xr, yr], axis=1)
            else:
                Xr, yr = self._internal_over(X, y)
                res = pd.concat([Xr, yr], axis=1)
        elif method == "RandomUnderSampler":
            if IMB:
                rus = RandomUnderSampler(random_state=42)
                Xr, yr = rus.fit_resample(X, y)
                res = pd.concat([Xr, yr], axis=1)
            else:
                Xr, yr = self._internal_under(X, y)
                res = pd.concat([Xr, yr], axis=1)
        else:  # SMOTE
            if not IMB:
                messagebox.showwarning("SMOTE indisponible","Installe imbalanced-learn pour SMOTE ou choisis une autre méthode.")
                return
            # SMOTE nécessite des features numériques (pas CleanText brut)
            num = X.select_dtypes(include="number")
            if num.shape[1]==0:
                messagebox.showwarning("Colonnes numériques requises",
                    "SMOTE nécessite des colonnes numériques. Vectorise d'abord le texte (ex: TF-IDF).")
                return
            sm = SMOTE(random_state=42)
            Xr, yr = sm.fit_resample(num, y)
            res = pd.concat([Xr, yr], axis=1)

        self.log("Après: " + str(Counter(res[ycol])))
        save = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="resampled.csv",
                                            filetypes=[("CSV","*.csv")])
        if not save: return
        res.to_csv(save, index=False, sep=";", encoding="utf-8-sig")
        self.log(f"OK: {save}")

if __name__ == "__main__":
    App().mainloop()
