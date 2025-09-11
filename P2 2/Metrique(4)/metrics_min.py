#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# --- Auto-install deps if missing ---
import sys, subprocess
def ensure(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for p in ["pandas","sklearn","tk"]:
    try:
        ensure(p if p!="sklearn" else "scikit-learn")
    except Exception:
        pass

# --- Imports ---
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    def clean(c):
        if isinstance(c, str):
            c = c.replace("\ufeff","").replace("\uFEFF","").strip().strip("'").strip('"')
        return c
    df = df.copy()
    df.columns = [clean(c) for c in df.columns]
    return df

class MetricsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Métriques (y_true / y_pred)")
        self.geometry("760x560")
        self.df = None

        root = ttk.Frame(self, padding=8); root.pack(fill="both", expand=True)

        # File
        ttk.Label(root, text="CSV :").grid(row=0, column=0, sticky="w")
        self.csv_var = tk.StringVar(value="(aucun)")
        ttk.Entry(root, textvariable=self.csv_var, width=70).grid(row=0, column=1, sticky="we")
        ttk.Button(root, text="Ouvrir...", command=self.open_csv).grid(row=0, column=2, padx=6)

        # Columns
        ttk.Label(root, text="y_true :").grid(row=1, column=0, sticky="w", pady=(8,0))
        self.true_var = tk.StringVar()
        self.true_cb = ttk.Combobox(root, textvariable=self.true_var, state="readonly", width=30)
        self.true_cb.grid(row=1, column=1, sticky="w", pady=(8,0))

        ttk.Label(root, text="y_pred :").grid(row=2, column=0, sticky="w")
        self.pred_var = tk.StringVar()
        self.pred_cb = ttk.Combobox(root, textvariable=self.pred_var, state="readonly", width=30)
        self.pred_cb.grid(row=2, column=1, sticky="w")

        # Averaging
        ttk.Label(root, text="Averaging :").grid(row=3, column=0, sticky="w", pady=(8,0))
        self.avg_var = tk.StringVar(value="macro")
        self.avg_cb = ttk.Combobox(root, textvariable=self.avg_var, state="readonly",
                                   values=["binary","micro","macro","weighted"], width=12)
        self.avg_cb.grid(row=3, column=1, sticky="w", pady=(8,0))

        # Actions
        btns = ttk.Frame(root); btns.grid(row=4, column=0, columnspan=3, pady=8, sticky="w")
        ttk.Button(btns, text="Calculer", command=self.compute).pack(side="left")
        ttk.Button(btns, text="Exporter rapport...", command=self.export_report).pack(side="left", padx=8)

        # Output
        self.out = tk.Text(root, height=22)
        self.out.grid(row=5, column=0, columnspan=3, sticky="nsew")
        root.columnconfigure(1, weight=1)
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
                df=None
                for sep in [";","\t",","]:
                    try:
                        df = pd.read_csv(p, sep=sep, dtype=str, encoding="utf-8")
                        break
                    except Exception:
                        pass
                if df is None: raise
            df = sanitize_columns(df)
            self.df = df
            self.csv_var.set(p)
            cols = list(df.columns)
            self.true_cb["values"] = cols
            self.pred_cb["values"] = cols
            # auto-guess
            for c in ["y_true","true","target","label","Sentiment","Rating"]:
                if c in cols: self.true_var.set(c); break
            for c in ["y_pred","pred","prediction","Prediction"]:
                if c in cols: self.pred_var.set(c); break
            self.out.delete("1.0","end")
            self.log(f"✅ CSV chargé: {len(df)} lignes, colonnes={cols}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Lecture CSV:\n{e}")

    def compute(self):
        if self.df is None:
            messagebox.showinfo("Info", "Charge d'abord un CSV.")
            return
        yt, yp = self.true_var.get(), self.pred_var.get()
        if yt=="" or yp=="":
            messagebox.showerror("Erreur","Choisis y_true et y_pred."); return
        y_true = self.df[yt].astype(str)
        y_pred = self.df[yp].astype(str)

        avg = self.avg_var.get()
        try:
            acc = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, average=avg if avg!="binary" else "binary", zero_division=0)
            rec  = recall_score(y_true, y_pred, average=avg if avg!="binary" else "binary", zero_division=0)
            f1m  = f1_score(y_true, y_pred, average="macro", zero_division=0)
            f1w  = f1_score(y_true, y_pred, average="weighted", zero_division=0)
            labels = sorted(set(y_true) | set(y_pred))
            cm   = confusion_matrix(y_true, y_pred, labels=labels)

            self.out.delete("1.0","end")
            self.log("=== Résultats ===")
            self.log(f"Accuracy : {acc:.4f}")
            self.log(f"Precision ({avg}) : {prec:.4f}")
            self.log(f"Recall    ({avg}) : {rec:.4f}")
            self.log(f"F1-macro  : {f1m:.4f}")
            self.log(f"F1-weighted : {f1w:.4f}")
            self.log("")
            self.log("Matrice de confusion (lignes = vrais, colonnes = prédits) :")
            header = "      " + " ".join([f"{str(l):>8}" for l in labels])
            self.log(header)
            for i, l in enumerate(labels):
                row = " ".join([f"{v:>8d}" for v in cm[i]])
                self.log(f"{str(l):>6} {row}")
            self.log("")
            self.log("Rapport de classification :")
            self.log(classification_report(y_true, y_pred, zero_division=0))
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec du calcul des métriques:\n{e}")

    def export_report(self):
        if self.df is None:
            messagebox.showinfo("Info", "Charge d'abord un CSV et calcule les métriques.")
            return
        # Reproduit un export simple (macro/weighted + cm) en CSV
        yt, yp = self.true_var.get(), self.pred_var.get()
        if yt=="" or yp=="":
            messagebox.showerror("Erreur","Choisis y_true et y_pred."); return
        y_true = self.df[yt].astype(str)
        y_pred = self.df[yp].astype(str)
        labels = sorted(set(y_true) | set(y_pred))
        cm   = confusion_matrix(y_true, y_pred, labels=labels)

        # Build a small dataframe summary
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        df_sum = pd.DataFrame({
            "metric":["accuracy","f1_macro","f1_weighted"],
            "value":[acc, f1m, f1w]
        })
        # Confusion matrix flattened
        rows = []
        for i, li in enumerate(labels):
            for j, lj in enumerate(labels):
                rows.append({"true":li, "pred":lj, "count":int(cm[i][j])})
        df_cm = pd.DataFrame(rows)

        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="metrics_summary.csv",
                                            filetypes=[("CSV","*.csv")])
        if not path: return
        try:
            with open(path, "w", encoding="utf-8-sig") as f:
                f.write("# summary\n")
            df_sum.to_csv(path, mode="a", index=False, encoding="utf-8-sig")
            with open(path, "a", encoding="utf-8-sig") as f:
                f.write("\n# confusion_matrix\n")
            df_cm.to_csv(path, mode="a", index=False, encoding="utf-8-sig")
            messagebox.showinfo("Export", f"✅ Rapport exporté: {path}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Echec export:\n{e}")

if __name__ == "__main__":
    MetricsApp().mainloop()
