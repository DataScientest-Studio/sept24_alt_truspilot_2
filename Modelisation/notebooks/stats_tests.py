# Modelisation/notebooks/stats_tests.py
# Usage: python Modelisation/notebooks/stats_tests.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

INPUT = "trustpilot_dataset_final_features.csv"   # ton CSV features
OUT_DIR = Path("Modelisation/reports")
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT = OUT_DIR / "stats_report.txt"

# ---------------- Utils ----------------
def cohen_d(x, y):
    x, y = np.array(x, float), np.array(y, float)
    nx, ny = len(x), len(y)
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    s = np.sqrt(((nx - 1)*vx + (ny - 1)*vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / s if s > 0 else np.nan

def cliffs_delta(x, y):
    # |delta| ~ 0.147 small, 0.33 medium, 0.474 large
    x, y = list(x), list(y)
    gt = sum(1 for xi in x for yi in y if xi > yi)
    lt = sum(1 for xi in x for yi in y if xi < yi)
    n = len(x)*len(y)
    return (gt - lt) / n if n else np.nan

def describe_series(s):
    return {
        "n": int(s.notna().sum()),
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)),
        "min": float(s.min()),
        "q25": float(s.quantile(0.25)),
        "median": float(s.median()),
        "q75": float(s.quantile(0.75)),
        "max": float(s.max()),
    }

# -------------- Main -------------------
df = pd.read_csv(INPUT)

# Créer un label binaire pos/neg si besoin (garde 3 comme neutre et on l’exclut)
df["label_bin"] = np.where(df["Rating"] >= 4, 1, np.where(df["Rating"] <= 2, 0, np.nan))
df_bin = df.dropna(subset=["label_bin"]).copy()
df_bin["label_bin"] = df_bin["label_bin"].astype(int)

# Variables numériques à tester (issues de ton feature_engineering)
num_cols = ["text_length", "word_count", "exclamation_count", "question_count",
            "capital_ratio", "polarity", "subjectivity"]

lines = []
lines.append("=== TESTS STATISTIQUES SUR NOUVELLES VARIABLES ===\n")
lines.append(f"Fichier: {INPUT}")
lines.append(f"Total rows: {len(df)} | Binaire (exclut Rating=3): {len(df_bin)}")
lines.append("Binaire: 0 = neg (1-2), 1 = pos (4-5)\n")

# 1) Comparaisons 0 vs 1 (Mann-Whitney + effet)
for col in num_cols:
    if col not in df_bin.columns:
        continue
    x = df_bin.loc[df_bin["label_bin"] == 0, col].dropna()
    y = df_bin.loc[df_bin["label_bin"] == 1, col].dropna()
    if len(x) < 10 or len(y) < 10:
        continue

    # Normalité rapide -> si non-normal, Mann-Whitney
    _, p_shapiro_x = stats.shapiro(x.sample(min(len(x), 500), random_state=0))
    _, p_shapiro_y = stats.shapiro(y.sample(min(len(y), 500), random_state=0))

    if p_shapiro_x < 0.05 or p_shapiro_y < 0.05:
        test_name = "Mann-Whitney U (non param.)"
        stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
        effect = cliffs_delta(x, y)
        effect_name = "Cliff's delta"
    else:
        # t-test de Welch
        test_name = "t-test Welch (param.)"
        stat, p = stats.ttest_ind(x, y, equal_var=False)
        effect = cohen_d(x, y)
        effect_name = "Cohen's d"

    dx, dy = describe_series(x), describe_series(y)
    lines.append(f"--- {col} ---")
    lines.append(f"neg (n={dx['n']}): mean={dx['mean']:.4f} (sd={dx['std']:.4f})")
    lines.append(f"pos (n={dy['n']}): mean={dy['mean']:.4f} (sd={dy['std']:.4f})")
    lines.append(f"Test: {test_name} -> stat={stat:.4f}, p-value={p:.3e}")
    lines.append(f"Effet: {effect_name} = {effect:.3f}\n")

# 2) Corrélations avec Rating (ordinal 1–5) -> Spearman
lines.append("\n=== Corrélations (Spearman) avec Rating (1–5) ===\n")
for col in num_cols:
    if col not in df.columns:
        continue
    s = df[col].astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    sub = df.loc[s.index]
    rho, p = stats.spearmanr(sub["Rating"], s)
    lines.append(f"{col}: rho={rho:.3f}, p={p:.3e}")

# Écriture
OUT_DIR.mkdir(parents=True, exist_ok=True)
with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))

print("✅ Rapport écrit dans:", REPORT)
print("\n".join(lines[:25]), "\n... (tronqué) ...")
