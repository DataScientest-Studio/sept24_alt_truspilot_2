import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

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