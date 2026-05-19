# Projet Trustpilot — Analyse et prédiction du sentiment des avis clients

## 🎯 Objectif du projet

Ce projet a pour objectif d’analyser des avis clients issus de la plateforme Trustpilot afin de :

- explorer et comprendre les données textuelles ;
- produire des visualisations descriptives (EDA) ;
- extraire des features pertinentes à partir du texte ;
- entraîner un modèle de classification binaire du sentiment ;
- exposer le modèle via une API FastAPI ;
- suivre les entraînements avec MLflow ;
- versionner les modèles avec MLflow Registry ;
- proposer une démonstration interactive via une application Streamlit.

Le projet s’inscrit dans le cadre de la formation Data Scientist — DataScientest.

---

## 🧠 Démarche Data Science et MLOps

La démarche suivie est structurée autour des étapes classiques d’un projet de Data Science, enrichie par une première couche MLOps :

1. Exploration des données
2. Préprocessing du texte
3. Feature engineering
4. Modélisation
5. Évaluation des performances
6. Démonstration via Streamlit
7. Exposition du modèle via une API FastAPI
8. Tests automatisés de l’API
9. Suivi des expériences avec MLflow
10. Versioning des modèles avec MLflow Registry
11. Comparaison automatique des versions et sélection du meilleur modèle

---

## 📁 Structure du projet

sept24_alt_truspilot_2/  
│  
├── Modelisation/  
│   ├── api.py                         API FastAPI de prédiction  
│   ├── training_mlflow.py             Script d’entraînement avec MLflow  
│   ├── tests/                         Tests automatisés de l’API  
│   ├── data/                          Base SQLite contenant les avis  
│   ├── dataviz/                       Visualisations EDA, wordclouds, distributions  
│   ├── models/                        Modèles sauvegardés localement  
│   ├── reports/                       Résultats et métriques  
│   ├── trustpilot_streamlit/          Application Streamlit de démonstration  
│   └── README_MLOPS.md                Documentation détaillée de la partie MLOps  
│  
├── Rapport exploration des données.xlsx  
├── Rendu_1.pdf  
├── Rendu_2.pdf  
├── jury_doc.txt  
├── projet_trustpilot_guide_pas_a_pas_version_simple.md  
│  
├── requirements.txt                   Dépendances Python globales  
├── README.md                          Présentation générale du projet  
└── LICENSE  

---

## 🧪 Installation & environnement

Installer les dépendances Python depuis la racine du projet :

    pip install -r requirements.txt

Si besoin, installer également les dépendances spécifiques à la partie modélisation :

    pip install -r Modelisation/requirements.txt

---

## 🚀 Lancer l’application Streamlit

Depuis la racine du projet :

    streamlit run Modelisation/trustpilot_streamlit/app.py

---

## ⚙️ Lancer l’API FastAPI

Depuis le dossier Modelisation :

    cd Modelisation
    python -m uvicorn api:app --reload --port 8000

L’API est disponible ici :

    http://127.0.0.1:8000

La documentation Swagger est disponible ici :

    http://127.0.0.1:8000/docs

---

## 🔎 Exemple de prédiction avec l’API

Exemple de requête PowerShell :

    Invoke-RestMethod `
      -Method POST `
      -Uri "http://127.0.0.1:8000/predict" `
      -Headers @{"Content-Type"="application/json"} `
      -Body '{"text":"This product is amazing and I love it"}'

Exemple de réponse :

    text                 : This product is amazing and I love it
    prediction           : 1
    label                : positive
    probability_negative : 0.0061
    probability_positive : 0.9938

---

## ✅ Tests automatisés

Depuis le dossier Modelisation :

    python -m pytest

Résultat obtenu :

    5 passed

Les tests vérifient notamment :

- la route `/` ;
- la route `/health` ;
- la route `/model-info` ;
- la route `/predict` ;
- le refus d’un texte vide par la validation Pydantic.

---

## 📊 Suivi d’expériences avec MLflow

Le script suivant permet d’entraîner le modèle avec suivi MLflow :

    Modelisation/training_mlflow.py

Depuis le dossier Modelisation :

    python training_mlflow.py

Ce script permet de :

- charger les données depuis la base SQLite ;
- préparer la cible binaire ;
- entraîner un pipeline TF-IDF + LogisticRegression ;
- logger les paramètres dans MLflow ;
- logger les métriques dans MLflow ;
- enregistrer le modèle dans MLflow Registry ;
- comparer le nouveau modèle avec la meilleure version précédente ;
- marquer la meilleure version avec l’alias `best`.

---

## 🧬 MLflow Registry

Le modèle est enregistré dans MLflow Registry sous le nom :

    trustpilot_sentiment_model

À chaque entraînement, une nouvelle version est créée.

Exemple observé :

    Version 1 -> alias best
    Version 2 -> pas d’alias

Cela signifie que la Version 2 a bien été créée, mais qu’elle n’a pas remplacé la Version 1 car elle n’était pas meilleure sur la métrique principale `f1_macro`.

La documentation détaillée de cette partie est disponible ici :

    Modelisation/README_MLOPS.md

---

## 📊 Résultats

Les principaux résultats et analyses sont disponibles dans :

- Rendu_1.pdf
- Rendu_2.pdf
- Rapport exploration des données.xlsx

Les visualisations générées lors de l’exploration sont regroupées dans :

- Modelisation/dataviz/

Les métriques de comparaison sont disponibles dans :

- Modelisation/reports/comparison_metrics.csv

---

## ⚠️ Limites du projet

- Dataset de taille limitée
- Avis issus d’une seule plateforme
- Modèle volontairement simple afin de rester interprétable
- Première version MLOps locale, non encore déployée en production
- Automatisation Airflow ou cron job prévue comme amélioration future
- Monitoring de dérive des données non encore implémenté

---

## 🔮 Améliorations futures

Plusieurs pistes d’amélioration sont prévues :

- charger directement dans l’API le modèle marqué `best` dans MLflow Registry ;
- automatiser l’entraînement avec Airflow ou un cron job ;
- ajouter un pipeline CI/CD ;
- ajouter davantage de tests unitaires et d’intégration ;
- monitorer les prédictions en production ;
- suivre la dérive des données ;
- comparer plusieurs familles de modèles ;
- améliorer la gestion des logs et des erreurs.

---

## 👤 Auteur

Pierre Poulouin  
Formation Data Scientist — DataScientest