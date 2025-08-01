import pandas as pd
import re

# === 1. Chargement des données ===
df = pd.read_csv("trustpilot_ringconn_v3.csv", quoting=1)

# === 2. Nettoyage des colonnes utiles ===
# Suppression des avis sans texte ou sans rating
df = df.dropna(subset=['Content', 'Rating'])

# Convertir Rating en numérique
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
df = df.dropna(subset=['Rating'])

# Optionnel : suppression des lignes avec Rating hors 1-5
df = df[df['Rating'].isin([1, 2, 3, 4, 5])]

# === 3. Fusion Title + Content ===
df['FullText'] = df[['Title', 'Content']].fillna("").agg(" ".join, axis=1)

# === 4. Nettoyage de texte ===
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)  # On garde uniquement lettres et espaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

df['CleanText'] = df['FullText'].apply(clean_text)

# === 5. Export des données nettoyées ===
df_cleaned = df[['CleanText', 'Rating']]
df_cleaned.to_csv("trustpilot_cleaned.csv", index=False, quoting=1)

print("✅ Dataset nettoyé exporté dans 'trustpilot_cleaned.csv'")
