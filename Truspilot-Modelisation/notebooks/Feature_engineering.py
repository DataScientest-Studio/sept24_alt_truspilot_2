import pandas as pd
import re
from textblob import TextBlob

# === Chargement du fichier brut ===
file_path = "trustpilot_ringconn_v3.csv"
df = pd.read_csv(file_path, quoting=1)

# === Prétraitement du texte pour CleanText ===
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", str(text))  # garder lettres et espaces
    text = text.lower()  # mettre en minuscules
    text = re.sub(r"\s+", " ", text).strip()  # retirer espaces multiples
    return text

# === Création de la colonne CleanText ===
df['Content'] = df['Content'].fillna("")
df['CleanText'] = df['Content'].apply(clean_text)

# === Création des features enrichies ===
df['text_length'] = df['CleanText'].apply(len)
df['word_count'] = df['CleanText'].apply(lambda x: len(x.split()))
df['exclamation_count'] = df['Content'].apply(lambda x: str(x).count('!'))
df['question_count'] = df['Content'].apply(lambda x: str(x).count('?'))
df['capital_ratio'] = df['Content'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1))
df['polarity'] = df['Content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
df['subjectivity'] = df['Content'].apply(lambda x: TextBlob(str(x)).sentiment.subjectivity)

# === Sauvegarde ===
output_path = "trustpilot_ringconn_features.csv"
df.to_csv(output_path, index=False, quoting=1)
print(f"✅ Fichier enrichi enregistré sous : {output_path}")