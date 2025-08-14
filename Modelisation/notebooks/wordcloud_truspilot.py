# notebooks/03_wordcloud_trustpilot.py
# Usage:
#   python notebooks/03_wordcloud_trustpilot.py --input notebooks/trustpilot_dataset_final_clean_regex.csv --outdir outputs/wordclouds
#   (ou remplace l’input par trustpilot_dataset_final_cleaned_advanced.csv si tu utilises la version avancée)

import argparse
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# WordCloud + stopwords NLTK (EN/FR)
from wordcloud import WordCloud
import nltk
from nltk.corpus import stopwords

# --------- Préparation ---------
def ensure_nltk():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")

def build_stopwords(extra_stop=None):
    """Stopwords bilingues + mots spécifiques au domaine Trustpilot."""
    ensure_nltk()
    sw = set(stopwords.words("english")) | set(stopwords.words("french"))

    domain_words = {
        # marques / produits / mots génériques peu informatifs
        "ring", "ringconn", "oura", "gen", "air", "sleep", "apnea", "device",
        "bagu", "bague", "anneau", "commande", "remboursement", "livraison",
        "customer", "service", "client", "clients", "company",
        "product", "products", "order", "ordered", "shipping",
        "app", "application", "support", "sizer",
        # fillers & tokens fréquents déjà nettoyés partiellement
        "one", "also", "use", "using"
    }

    # on passe tout en minuscule pour cohérence
    sw = {w.lower() for w in sw} | {w.lower() for w in domain_words}

    if extra_stop:
        sw |= {w.lower() for w in extra_stop}

    return sw

def normalize(text: str) -> str:
    """Nettoyage léger pour la dataviz (garde chiffres)."""
    if not isinstance(text, str):
        return ""
    t = text.lower()
    # remplace ponctuation par espace (au cas où)
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    # espaces multiples
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokens_from_series(series: pd.Series, sw: set) -> list:
    """Concatène, normalise et filtre les stopwords → liste de tokens."""
    text = " ".join(series.dropna().astype(str).tolist())
    text = normalize(text)
    tokens = [w for w in text.split() if w not in sw and len(w) > 2]
    return tokens

def most_common(tokens, n=20):
    from collections import Counter
    c = Counter(tokens)
    return c.most_common(n)

def top_bigrams(tokens, n=20, min_count=2):
    from collections import Counter
    bigs = Counter(zip(tokens[:-1], tokens[1:]))
    # on filtre les bigrams trop rares
    items = [(f"{a} {b}", cnt) for (a, b), cnt in bigs.items() if cnt >= min_count]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:n]

def plot_wordcloud(tokens, outpath, background_color="white", max_words=150, max_font_size=70):
    txt = " ".join(tokens)
    wc = WordCloud(
        width=1200, height=700,
        background_color=background_color,
        stopwords=set(),  # déjà filtré
        max_words=max_words,
        max_font_size=max_font_size,
        random_state=42
    )
    wc.generate(txt)
    plt.figure(figsize=(12,7))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def plot_bar(items, title, outpath, rotation=45):
    labels = [w for w, _ in items]
    counts = [c for _, c in items]
    plt.figure(figsize=(12,6))
    plt.bar(range(len(labels)), counts)
    plt.xticks(range(len(labels)), labels, rotation=rotation, ha="right")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180)
    plt.close()

def main(input_path, outdir):
    os.makedirs(outdir, exist_ok=True)
    df = pd.read_csv(input_path)
    # On prend la colonne la plus propre disponible
    text_col = (
        "CleanText_regex" if "CleanText_regex" in df.columns else
        "CleanText" if "CleanText" in df.columns else
        "FullText" if "FullText" in df.columns else
        "Content"
    )
    if text_col not in df.columns:
        raise ValueError("Aucune colonne texte trouvée (CleanText_regex / CleanText / FullText / Content).")

    if "Rating" not in df.columns:
        raise ValueError("La colonne 'Rating' est nécessaire pour séparer pos/neg.")

    sw = build_stopwords()

    # Découpage par sentiment (seuils simples)
    df_pos = df[df["Rating"] >= 4]
    df_neg = df[df["Rating"] <= 2]

    # Tokens
    tokens_all = tokens_from_series(df[text_col], sw)
    tokens_pos = tokens_from_series(df_pos[text_col], sw)
    tokens_neg = tokens_from_series(df_neg[text_col], sw)

    # --- WordClouds ---
    plot_wordcloud(tokens_all, os.path.join(outdir, "wc_all.png"))
    plot_wordcloud(tokens_pos, os.path.join(outdir, "wc_positive.png"), background_color="white")
    plot_wordcloud(tokens_neg, os.path.join(outdir, "wc_negative.png"), background_color="white")

    # --- Histogrammes unigrams ---
    top_all = most_common(tokens_all, n=20)
    top_pos = most_common(tokens_pos, n=20)
    top_neg = most_common(tokens_neg, n=20)

    plot_bar(top_all, "Top 20 mots (tous avis)", os.path.join(outdir, "hist_all_top20.png"))
    plot_bar(top_pos, "Top 20 mots (avis positifs)", os.path.join(outdir, "hist_pos_top20.png"))
    plot_bar(top_neg, "Top 20 mots (avis négatifs)", os.path.join(outdir, "hist_neg_top20.png"))

    # --- Histogrammes bigrams (optionnel mais utile) ---
    big_all = top_bigrams(tokens_all, n=20, min_count=2)
    big_pos = top_bigrams(tokens_pos, n=20, min_count=2)
    big_neg = top_bigrams(tokens_neg, n=20, min_count=2)

    plot_bar(big_all, "Top 20 bigrams (tous avis)", os.path.join(outdir, "bigrams_all_top20.png"))
    plot_bar(big_pos, "Top 20 bigrams (avis positifs)", os.path.join(outdir, "bigrams_pos_top20.png"))
    plot_bar(big_neg, "Top 20 bigrams (avis négatifs)", os.path.join(outdir, "bigrams_neg_top20.png"))

    print("[OK] Wordclouds & histogrammes générés dans:", outdir)
    print(" - wc_all.png / wc_positive.png / wc_negative.png")
    print(" - hist_all_top20.png / hist_pos_top20.png / hist_neg_top20.png")
    print(" - bigrams_all_top20.png / bigrams_pos_top20.png / bigrams_neg_top20.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prétraitement avancé du texte Trustpilot")
    parser.add_argument("--input", type=str, default="trustpilot_dataset_final_cleaned.csv", help="Fichier CSV en entrée")
    parser.add_argument("--output", type=str, default="Modelisation/Dataviz/wordclouds", help="dossier de sortie images")
    args = parser.parse_args()

    main(args.input, args.output)

