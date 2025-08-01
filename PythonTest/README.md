# Projet d'Analyse de Sentiment

Ce projet permet d'analyser les sentiments des avis clients en français en utilisant des techniques de machine learning.

## 📁 Structure du projet

```
PythonTest/
├── Scrapper(1)/          # Scraping des avis Trustpilot
├── Clean(2)/             # Nettoyage des données
├── Prepross(3)/          # Prétraitement avancé
├── Graph(4)/             # Génération de graphiques
├── Baseline(5)/          # Modèles de base
├── Prediction(6)/        # Prédictions sur nouveaux avis
├── requirements.txt       # Dépendances
├── install_dependencies.py # Script d'installation
└── README.md             # Ce fichier
```

## 🚀 Installation

### Option 1 : Installation automatique
```bash
python install_dependencies.py
```

### Option 2 : Installation manuelle
```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
```

## 📋 Utilisation

### 1. Scraping des données (Scrapper(1)/scrap.py)
```bash
python "Scrapper(1)/scrap.py"
```
- Scrape les avis Trustpilot
- Génère un fichier CSV avec les données brutes

### 2. Nettoyage des données (Clean(2)/cleanData.py)
```bash
python "Clean(2)/cleanData.py"
```
- Interface graphique pour nettoyer les données
- Supprime les stopwords et normalise le texte

### 3. Prétraitement avancé (Prepross(3)/preProcData.py)
```bash
python "Prepross(3)/preProcData.py"
```
- Prétraitement avancé avec spaCy
- Lemmatisation et stemming
- Génération des étiquettes de sentiment

### 4. Modèles de base (Baseline(5)/baselienModele.py)
```bash
python "Baseline(5)/baselienModele.py"
```
- Entraîne des modèles de classification
- Sauvegarde les modèles dans le dossier `modeles/`

### 5. Prédictions (Prediction(6)/prediction.py)
```bash
python "Prediction(6)/prediction.py"
```
- Interface pour prédire les sentiments sur de nouveaux avis
- Utilise les modèles entraînés

## 🔧 Dépendances

- **pandas** : Manipulation des données
- **scikit-learn** : Machine learning
- **nltk** : Traitement du langage naturel
- **spacy** : NLP avancé
- **beautifulsoup4** : Scraping web
- **requests** : Requêtes HTTP
- **joblib** : Sauvegarde des modèles

## 📊 Formats de fichiers

Les fichiers CSV utilisent :
- Encodage : UTF-8 avec BOM
- Séparateur : Point-virgule (;)
- Colonnes attendues : Auteur, Date, Texte, Note

## 🐛 Résolution des problèmes

### Erreur "ModuleNotFoundError"
```bash
pip install pandas scikit-learn joblib nltk spacy beautifulsoup4 requests
```

### Erreur avec spaCy
```bash
python -m spacy download fr_core_news_sm
```

### Conflit de versions numpy
```bash
pip install --upgrade numpy==1.26.4
```

## 📝 Notes

- Assurez-vous d'utiliser l'environnement conda (base) pour éviter les conflits
- Les modèles sont sauvegardés dans le dossier `modeles/`
- Les résultats sont exportés en CSV avec l'encodage UTF-8-sig 