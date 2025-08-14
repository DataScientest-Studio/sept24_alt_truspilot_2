import pandas as pd
from textblob import TextBlob
from pathlib import Path

# Création auto des dossiers pipeline
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)

# === Chargement du fichier nettoyé ===
file_path = "trustpilot_dataset_final_cleaned.csv"  # Fichier issu de preprocessing_advanced
df = pd.read_csv(file_path, quoting=1)

# Vérification que CleanText existe
if 'CleanText' not in df.columns:
    raise ValueError("❌ La colonne 'CleanText' est introuvable dans le fichier d'entrée.")

# === Feature Engineering à partir de CleanText ===
df['text_length'] = df['CleanText'].apply(len)
df['word_count'] = df['CleanText'].apply(lambda x: len(str(x).split()))
df['exclamation_count'] = df['CleanText'].apply(lambda x: str(x).count('!'))
df['question_count'] = df['CleanText'].apply(lambda x: str(x).count('?'))
df['capital_ratio'] = df['CleanText'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
df['polarity'] = df['CleanText'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['subjectivity'] = df['CleanText'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# === Sauvegarde ===
output_path = "trustpilot_dataset_final_features.csv"
df.to_csv(output_path, index=False, quoting=1)

print(f"✅ Fichier enrichi enregistré sous : {output_path}")
