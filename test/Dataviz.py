import sys
import os

# Vérification de l'environnement virtuel
if sys.prefix == sys.base_prefix:
    print("\033[91m[ERREUR] Veuillez activer l'environnement virtuel avant de lancer ce script :\033[0m")
    print("  source venv/bin/activate && python notebooks/Dataviz.py")
    sys.exit(1)

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import seaborn as sns

# Chargement des données
file_path = 'trustpilot_ringconn_v3.csv'
df = pd.read_csv(file_path, quoting=1)
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df['Content_length'] = df['Content'].apply(lambda x: len(str(x)))

# 1 - Distribution de la longueur des avis
plt.figure(figsize=(8,5))
df['Content_length'].hist(bins=50)
plt.title('Distribution de la longueur des avis')
plt.xlabel('Longueur du contenu (en caractères)')
plt.ylabel("Nombre d'avis")
plt.grid(True)
plt.savefig("distribution_longueur_avis.png")
plt.close()

# 2 - Nombre d'avis par jour
df['Date_only'] = df['Date'].dt.date
avis_par_jour = df.groupby('Date_only').size()

plt.figure(figsize=(12,6))
avis_par_jour.plot()
plt.title("Nombre d'avis par jour")
plt.xlabel("Date")
plt.ylabel("Nombre d'avis")
plt.grid(True)
plt.savefig("nombre_avis_par_jour.png")
plt.close()

# 3 - Nombre d'avis par mois
df['YearMonth'] = df['Date'].dt.to_period('M')
avis_par_mois = df.groupby('YearMonth').size()

plt.figure(figsize=(12,6))
avis_par_mois.plot(kind='bar')
plt.title("Nombre d'avis par mois")
plt.xlabel("Mois")
plt.ylabel("Nombre d'avis")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("nombre_avis_par_mois.png")
plt.close()

# 4 - WordCloud (version simplifiée sans nltk)
text_content = " ".join(df['Content'].dropna().tolist())
text_content_clean = re.sub(r"[^A-Za-z\s]", " ", text_content).lower()

manual_stopwords = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'i', 'is', 'it', 'this', 'for', 'that',
    'on', 'with', 'was', 'my', 'have', 'but', 'at', 'they', 'are', 'so', 'very',
    'not', 'be', 'had', 'as', 'you', 'from', 'we', 'if', 'or', 'by', 'an', 'all',
    'has', 'our', 'more', 'just', 'me', 'can', 'their', 'were', 'after', 'about',
    'would', 'when', 'which', 'what', 'them', 'who', 'get', 'out', 'only', 'also',
    'because', 'how', 'could', 'been', 'still', 'too', 'am', 'got', 'he', 'she', 'will'
])

wordcloud = WordCloud(width=1200, height=600, background_color='white',
                      stopwords=manual_stopwords, collocations=False).generate(text_content_clean)

plt.figure(figsize=(15, 7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title("WordCloud des mots les plus fréquents dans les avis", fontsize=16)
plt.savefig("wordcloud.png")
plt.close()

# 5 - Distribution de la longueur des titres
df['Title_length'] = df['Title'].fillna('').apply(len)

plt.figure(figsize=(8,5))
df['Title_length'].hist(bins=30, color='orange')
plt.title('Distribution de la longueur des titres des avis')
plt.xlabel('Longueur du titre (en caractères)')
plt.ylabel("Nombre d'avis")
plt.grid(True)
plt.savefig("distribution_longueur_titres.png")
plt.close()

# --- 1. Distribution des notes (Ratings) ---
plt.figure(figsize=(6,4))
sns.countplot(x='Rating', data=df, palette='viridis')
plt.title('Distribution des notes')
plt.xlabel('Note')
plt.ylabel('Nombre d\'avis')
plt.savefig("distribution_notes.png")
plt.close()

# --- 2. Nombre d'avis par mois ---
df['YearMonth'] = df['Date'].dt.to_period('M')
plt.figure(figsize=(10,4))
df.groupby('YearMonth').size().plot()
plt.title('Nombre d\'avis par mois')
plt.xlabel('Mois')
plt.ylabel('Nombre d\'avis')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("nombre_avis_par_mois.png")
plt.close()

# --- 3. Note moyenne par mois ---
plt.figure(figsize=(10,4))
df.groupby('YearMonth')['Rating'].mean().plot()
plt.title('Note moyenne par mois')
plt.xlabel('Mois')
plt.ylabel('Note moyenne')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("note_moyenne_par_mois.png")
plt.close()

# --- 4. Nombre d'avis par pays ---
if 'Country' in df.columns:
    plt.figure(figsize=(10,4))
    df['Country'].value_counts().plot(kind='bar')
    plt.title('Nombre d\'avis par pays')
    plt.xlabel('Pays')
    plt.ylabel('Nombre d\'avis')
    plt.tight_layout()
    plt.savefig("nombre_avis_par_pays.png")
    plt.close()

# --- 5. Nombre d'avis laissés par utilisateur ---
if 'Author' in df.columns:
    plt.figure(figsize=(10,4))
    df['Author'].value_counts().head(20).plot(kind='bar')
    plt.title('Top 20 des auteurs les plus actifs')
    plt.xlabel('Auteur')
    plt.ylabel('Nombre d\'avis laissés')
    plt.tight_layout()
    plt.savefig("top20_auteurs_actifs.png")
    plt.close()

# --- 6. Corrélation activité/note ---
if 'ReviewsCount' in df.columns:
    plt.figure(figsize=(6,4))
    plt.scatter(df['ReviewsCount'], df['Rating'], alpha=0.5)
    plt.title('Nombre de commentaires laissés vs Note')
    plt.xlabel('Nombre de commentaires laissés')
    plt.ylabel('Note')
    plt.tight_layout()
    plt.savefig("correlation_commentaires_note.png")
    plt.close()

# --- 7. Nombre d'avis par jour de la semaine ---
df['Weekday'] = df['Date'].dt.day_name()
plt.figure(figsize=(8,4))
df['Weekday'].value_counts().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]).plot(kind='bar')
plt.title('Nombre d\'avis par jour de la semaine')
plt.xlabel('Jour de la semaine')
plt.ylabel('Nombre d\'avis')
plt.tight_layout()
plt.savefig("nombre_avis_par_jour_semaine.png")
plt.close()

# --- 8. Longueur des avis ---
df['ContentLength'] = df['Content'].fillna('').apply(len)
plt.figure(figsize=(8,4))
plt.hist(df['ContentLength'], bins=30, color='skyblue')
plt.title('Longueur des avis (nombre de caractères)')
plt.xlabel('Nombre de caractères')
plt.ylabel('Nombre d\'avis')
plt.tight_layout()
plt.savefig("longueur_avis.png")
plt.close()