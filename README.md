# Projet Trustpilot — Analyse et prédiction du sentiment des avis clients

## 🎯 Objectif du projet
Ce projet a pour objectif d’analyser des avis clients issus de la plateforme Trustpilot afin de :
- explorer et comprendre les données textuelles,
- produire des visualisations descriptives (EDA),
- extraire des features pertinentes à partir du texte,
- entraîner un modèle de classification binaire du sentiment,
- proposer une démonstration interactive via une application Streamlit.

Le projet s’inscrit dans le cadre de la formation Data Scientist – DataScientest.

---

## 🧠 Démarche Data Science
La démarche suivie est structurée autour des étapes classiques d’un projet de data science :
1. Exploration des données  
2. Préprocessing du texte  
3. Feature engineering  
4. Modélisation  
5. Évaluation des performances  
6. Démonstration via Streamlit  

---

## 📁 Structure du projet

sept24_alt_truspilot_2/  
│  
├── Modelisation/  
│   ├── dataviz/                     Visualisations (EDA, wordclouds, distributions)  
│   ├── models/                      Résultats de modélisation (métriques, modèles)  
│   └── streamlit/                   Application Streamlit de démonstration  
│  
├── Rapport exploration des données.xlsx  
├── Rendu_1.pdf  
├── Rendu_2.pdf  
├── jury_doc.txt  
├── projet_trustpilot_guide_pas_a_pas_version_simple.md  
│  
├── requirements.txt                 Dépendances Python  
├── README.md  
└── LICENSE  

---

## 🧪 Installation & environnement
Installer les dépendances Python :

pip install -r requirements.txt

---

## 🚀 Lancer l’application Streamlit
Depuis la racine du projet :

streamlit run Modelisation/streamlit/app.py

---

## 📊 Résultats
Les principaux résultats et analyses sont disponibles dans :
- Rendu_1.pdf  
- Rendu_2.pdf  
- Rapport exploration des données.xlsx  

Les visualisations générées lors de l’exploration sont regroupées dans :
- Modelisation/dataviz/

---

## ⚠️ Limites du projet
- Dataset de taille limitée  
- Avis issus d’une seule plateforme  
- Modèle volontairement simple afin de rester interprétable  

---

## 👤 Auteur
Pierre Poulouin  
Formation Data Scientist — DataScientest
