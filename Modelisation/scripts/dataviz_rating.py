import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

# Chargement des données
file_path = 'trustpilot_ringconn_v3.csv'
df = pd.read_csv(file_path, quoting=1)
df['Content'] = df['Content'].fillna("")
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')

# Nettoyage des textes
manual_stopwords = set([
    'the', 'and', 'to', 'of', 'a', 'in', 'i', 'is', 'it', 'this', 'for', 'that',
    'on', 'with', 'was', 'my', 'have', 'but', 'at', 'they', 'are', 'so', 'very',
    'not', 'be', 'had', 'as', 'you', 'from', 'we', 'if', 'or', 'by', 'an', 'all',
    'has', 'our', 'more', 'just', 'me', 'can', 'their', 'were', 'after', 'about',
    'would', 'when', 'which', 'what', 'them', 'who', 'get', 'out', 'only', 'also',
    'because', 'how', 'could', 'been', 'still', 'too', 'am', 'got', 'he', 'she', 'will'
])

# Création des WordClouds pour les avis 5 étoiles
df_5 = df[df['Rating'] == 5]
text_5 = " ".join(df_5['Content'].tolist())
text_5 = re.sub(r"[^A-Za-z\s]", " ", text_5).lower()

wc_5 = WordCloud(width=1200, height=600, background_color='white',
                 stopwords=manual_stopwords, collocations=False).generate(text_5)

plt.figure(figsize=(15, 7))
plt.imshow(wc_5, interpolation='bilinear')
plt.axis('off')
plt.title("Mots les plus fréquents dans les avis 5 étoiles")
plt.savefig("wordcloud_5_etoiles.png")
plt.close()

# Création des WordClouds pour les avis 1 étoile
df_1 = df[df['Rating'] == 1]
text_1 = " ".join(df_1['Content'].tolist())
text_1 = re.sub(r"[^A-Za-z\s]", " ", text_1).lower()

wc_1 = WordCloud(width=1200, height=600, background_color='white',
                 stopwords=manual_stopwords, collocations=False).generate(text_1)

plt.figure(figsize=(15, 7))
plt.imshow(wc_1, interpolation='bilinear')
plt.axis('off')
plt.title("Mots les plus fréquents dans les avis 1 étoile")
plt.savefig("wordcloud_1_etoile.png")
plt.close()

print("✅ WordClouds enregistrés : 'wordcloud_5_etoiles.png' et 'wordcloud_1_etoile.png'")