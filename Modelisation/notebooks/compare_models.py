import json
from pathlib import Path
import pandas as pd

"""
Compare automatiquement les métriques de :
- Baseline (TF-IDF + ROS) : models/baseline/baseline_metrics.csv
- GridSearch LogReg       : models/gridsearch_logreg/test_metrics.json (si présent)
- TF-IDF + Sentiment      : models/sentiment_logreg/metrics_sentiment.json (si présent)

Sortie :
- Affichage console d’un tableau comparatif
- Écriture d’un CSV : reports/comparison_metrics.csv  (créé s’il n’existe pas)
"""

ROOT = Path(__file__).resolve().parents[1]  # notebooks/.. = racine projet

# chemins d'entrée possibles
BASELINE_CSV = ROOT / "models" / "baseline" / "baseline_metrics.csv"
GRIDSEARCH_JSON = ROOT / "models" / "gridsearch_logreg" / "test_metrics.json"
SENTIMENT_JSON = ROOT / "models" / "sentiment_logreg" / "metrics_sentiment.json"

# dossier de sortie
OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "comparison_metrics.csv"

rows = []

# 1) Baseline (on prend soit la meilleure ligne du CSV, soit la LogReg si tu préfères)
if BASELINE_CSV.exists():
    df_base = pd.read_csv(BASELINE_CSV)
    # tri par f1_macro décroissant
    df_base = df_base.sort_values("f1_macro", ascending=False)
    best_row = df_base.iloc[0].to_dict()
    rows.append({
        "model": f"Baseline ({best_row.get('model','?')})",
        "accuracy": best_row.get("accuracy", None),
        "balanced_accuracy": best_row.get("balanced_accuracy", None),
        "f1_macro": best_row.get("f1_macro", None),
    })
else:
    print(f"⚠️ Baseline non trouvée : {BASELINE_CSV}")

# 2) GridSearch LogReg (optionnel)
if GRIDSEARCH_JSON.exists():
    with open(GRIDSEARCH_JSON, "r") as f:
        m = json.load(f)
    rows.append({
        "model": "GridSearch (LogReg)",
        "accuracy": m.get("accuracy"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "f1_macro": m.get("f1_macro"),
    })
else:
    print(f"ℹ️ GridSearch non trouvé (facultatif) : {GRIDSEARCH_JSON}")

# 3) TF-IDF + Sentiment (optionnel)
if SENTIMENT_JSON.exists():
    with open(SENTIMENT_JSON, "r") as f:
        m = json.load(f)
    rows.append({
        "model": "TF-IDF + Sentiment (LogReg)",
        "accuracy": m.get("accuracy"),
        "balanced_accuracy": m.get("balanced_accuracy"),
        "f1_macro": m.get("f1_macro"),
    })
else:
    print(f"ℹ️ Modèle Sentiment non trouvé (facultatif) : {SENTIMENT_JSON}")

# 4) Tableau comparatif
if rows:
    cmp_df = pd.DataFrame(rows)
    # ordonner colonnes
    cmp_df = cmp_df[["model", "accuracy", "balanced_accuracy", "f1_macro"]]
    # trier par F1-macro décroissant
    cmp_df = cmp_df.sort_values("f1_macro", ascending=False)
    print("\n📊 Comparatif modèles :")
    print(cmp_df.to_string(index=False))

    # Sauvegarde
    cmp_df.to_csv(OUT_CSV, index=False)
    print(f"\n✅ Écrit : {OUT_CSV}")
else:
    print("❌ Aucun métrique trouvé. Lance d’abord l’entraînement des modèles.")
