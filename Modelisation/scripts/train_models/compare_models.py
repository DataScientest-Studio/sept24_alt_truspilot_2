import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]  # notebooks/.. = racine
BASELINE_CSV = ROOT / "models" / "baseline" / "baseline_metrics.csv"
GRIDSEARCH_JSON = ROOT / "models" / "gridsearch_logreg" / "test_metrics.json"
SENTIMENT_JSON = ROOT / "models" / "sentiment_logreg" / "metrics_sentiment.json"
SENTIMENT_TFIDF_JSON = ROOT / "models" / "sentiment_logreg_paramsTF_IDF" / "metrics_sentiment.json"

OUT_DIR = ROOT / "reports"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "comparison_metrics.csv"
OUT_MD  = OUT_DIR / "comparison_metrics.md"

rows = []

# 1) Baseline
if BASELINE_CSV.exists():
    df_base = pd.read_csv(BASELINE_CSV).sort_values("f1_macro", ascending=False)
    best_row = df_base.iloc[0].to_dict()
    baseline_f1 = float(best_row.get("f1_macro", 0))
    rows.append({
        "model": f"Baseline ({best_row.get('model','?')})",
        "accuracy": float(best_row.get("accuracy", 0)),
        "balanced_accuracy": float(best_row.get("balanced_accuracy", 0)),
        "f1_macro": baseline_f1,
        "delta_f1_vs_baseline": 0.0,
    })
else:
    print(f"⚠️ Baseline non trouvée : {BASELINE_CSV}")
    baseline_f1 = None

# 2) GridSearch LogReg
if GRIDSEARCH_JSON.exists():
    with open(GRIDSEARCH_JSON) as f:
        m = json.load(f)
    f1 = float(m.get("f1_macro", 0))
    rows.append({
        "model": "GridSearch (LogReg)",
        "accuracy": float(m.get("accuracy", 0)),
        "balanced_accuracy": float(m.get("balanced_accuracy", 0)),
        "f1_macro": f1,
        "delta_f1_vs_baseline": (None if baseline_f1 is None else f1 - baseline_f1),
    })
else:
    print(f"ℹ️ GridSearch non trouvé (facultatif) : {GRIDSEARCH_JSON}")

# 3) TF-IDF + Sentiment
if SENTIMENT_JSON.exists():
    with open(SENTIMENT_JSON) as f:
        m = json.load(f)
    f1 = float(m.get("f1_macro", 0))
    rows.append({
        "model": "TF-IDF + Sentiment (LogReg)",
        "accuracy": float(m.get("accuracy", 0)),
        "balanced_accuracy": float(m.get("balanced_accuracy", 0)),
        "f1_macro": f1,
        "delta_f1_vs_baseline": (None if baseline_f1 is None else f1 - baseline_f1),
    })
else:
    print(f"ℹ️ Modèle Sentiment non trouvé (facultatif) : {SENTIMENT_JSON}")

# 4) TF-IDF + Sentiment + Tweaks
if SENTIMENT_TFIDF_JSON.exists():
    with open(SENTIMENT_TFIDF_JSON) as f:
        m = json.load(f)
    f1 = float(m.get("f1_macro", 0))
    rows.append({
        "model": "TF-IDF + Sentiment + Tweaks",
        "accuracy": float(m.get("accuracy", 0)),
        "balanced_accuracy": float(m.get("balanced_accuracy", 0)),
        "f1_macro": f1,
        "delta_f1_vs_baseline": (None if baseline_f1 is None else f1 - baseline_f1),
    })
else:
    print(f"ℹ️ Modèle Sentiment + Tweaks non trouvé (facultatif) : {SENTIMENT_TFIDF_JSON}")

# 5) Sortie
if rows:
    cmp_df = pd.DataFrame(rows)[["model","accuracy","balanced_accuracy","f1_macro","delta_f1_vs_baseline"]]
    # tri par F1 décroissant
    cmp_df = cmp_df.sort_values("f1_macro", ascending=False)
    # arrondis jolis
    for col in ["accuracy","balanced_accuracy","f1_macro","delta_f1_vs_baseline"]:
        cmp_df[col] = cmp_df[col].astype(float).round(3)

    print("\n📊 Comparatif modèles :")
    print(cmp_df.to_string(index=False))

    # CSV
    cmp_df.to_csv(OUT_CSV, index=False)
    # Markdown (prêt à coller dans README/rapport)
    OUT_MD.write_text(
        "| Model | Accuracy | Balanced acc. | F1-macro | Δ F1 vs baseline |\n"
        "|---|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {r.model} | {r.accuracy} | {r.balanced_accuracy} | {r.f1_macro} | {'' if pd.isna(r.delta_f1_vs_baseline) else r.delta_f1_vs_baseline} |"
            for r in cmp_df.itertuples(index=False)
        ),
        encoding="utf-8"
    )

    print(f"\n✅ Écrit : {OUT_CSV}")
    print(f"✅ Écrit : {OUT_MD}")
else:
    print("❌ Aucun métrique trouvé. Lance d’abord l’entraînement des modèles.")
