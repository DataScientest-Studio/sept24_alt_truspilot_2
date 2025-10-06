# Evaluation simple d'un modèle avec F1-score (5 classes ou plus)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1) Charger le fichier CSV
df = pd.read_csv("data_oversampled.csv")   # Utiliser le fichier avec plus d'échantillons
X = df["CleanText"].astype(str)               # colonne texte
y = df["Rating"].astype(str)                  # colonne label

print(f"📊 Données chargées: {len(df)} échantillons")
print(f"🏷️  Classes présentes: {sorted(y.unique())}")
print(f"📈 Distribution des classes:")
print(y.value_counts().sort_index())

# Vérifier qu'on a assez d'échantillons par classe
min_samples_per_class = y.value_counts().min()
print(f"🔍 Minimum d'échantillons par classe: {min_samples_per_class}")

if min_samples_per_class < 2:
    print("❌ Erreur: Il faut au moins 2 échantillons par classe pour train_test_split avec stratify")
    print("💡 Solution: Utilisez le fichier data_oversampled.csv qui a plus d'échantillons")
    exit()

# 2) Séparer en train (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📚 Données d'entraînement: {len(X_train)} échantillons")
print(f"🧪 Données de test: {len(X_test)} échantillons")

# 3) Transformer le texte en vecteurs TF-IDF
print("🔄 Transformation TF-IDF en cours...")
vectorizer = TfidfVectorizer(max_features=1000, stop_words=None)  # Limiter les features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"📊 Features TF-IDF: {X_train_tfidf.shape[1]} dimensions")

# 4) Entraîner un modèle simple (régression logistique)
print("🤖 Entraînement du modèle...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_tfidf, y_train)
print("✅ Modèle entraîné!")

# 5) Prédire sur le test
print("🔮 Prédictions en cours...")
y_pred = model.predict(X_test_tfidf)

# 6) Évaluer avec F1-score
print("\n" + "="*60)
print("📈 RÉSULTATS D'ÉVALUATION")
print("="*60)

# F1-score macro (moyenne des F1-scores de chaque classe)
f1_macro = f1_score(y_test, y_pred, average="macro")
print(f"🎯 F1-score macro (moyenne): {f1_macro:.3f}")

# F1-score micro (calculé sur tous les échantillons)
f1_micro = f1_score(y_test, y_pred, average="micro")
print(f"🎯 F1-score micro (global): {f1_micro:.3f}")

# F1-score par classe
f1_per_class = f1_score(y_test, y_pred, average=None)
classes = sorted(y_test.unique())
print(f"\n📊 F1-score par classe:")
for i, class_name in enumerate(classes):
    print(f"   Classe {class_name}: {f1_per_class[i]:.3f}")

# Rapport complet (precision, recall, F1 par classe)
print(f"\n📋 Rapport de classification détaillé:")
print(classification_report(y_test, y_pred))

# Matrice de confusion
print(f"\n🎯 Matrice de confusion:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Créer une visualisation de la matrice de confusion
plt.figure(figsize=(10, 8))
sns.heatmap(cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=sorted(y_test.unique()),
            yticklabels=sorted(y_test.unique()),
            cbar_kws={'label': 'Nombre d\'échantillons'})

plt.title('Matrice de Confusion - Modèle de Classification', fontsize=16, fontweight='bold')
plt.xlabel('Prédictions', fontsize=12)
plt.ylabel('Vraies valeurs', fontsize=12)
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.tight_layout()

# Sauvegarder le graphique
plt.savefig('confusion_matrix_model.png', dpi=300, bbox_inches='tight')
print("📊 Matrice de confusion sauvegardée: confusion_matrix_model.png")

plt.show()

print("\n" + "="*60)
print("✅ ÉVALUATION TERMINÉE")
print("="*60)