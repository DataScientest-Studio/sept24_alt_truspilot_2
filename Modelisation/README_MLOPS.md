# Projet Trustpilot - Partie MLOps

## 1. Contexte du projet

Ce projet a pour objectif de prédire automatiquement le sentiment d’un avis client Trustpilot à partir de son texte.

Le problème a été simplifié en classification binaire :

- 0 : avis négatif ou neutre
- 1 : avis positif

La cible est construite à partir de la note Trustpilot :

    target = 1 if Rating >= 4 else 0

L’objectif principal de cette étape n’est pas uniquement d’obtenir le meilleur score possible, mais de mettre en place une chaîne MLOps simple permettant :

- d’exposer le modèle via une API FastAPI ;
- de tester les routes principales de l’API ;
- de suivre les entraînements avec MLflow ;
- de versionner les modèles avec MLflow Registry ;
- de comparer automatiquement les performances entre plusieurs versions ;
- de marquer la meilleure version du modèle avec l’alias best.

---

## 2. Structure utile du projet

    Modelisation/
    ├── api.py
    ├── training_mlflow.py
    ├── requirements.txt
    ├── data/
    │   └── trustpilot.db
    ├── models/
    │   └── trustpilot_logistic_tfidf.joblib
    ├── tests/
    │   └── test_api.py
    ├── mlruns/
    └── reports/

---

## 3. API FastAPI

L’API est définie dans le fichier :

    api.py

Elle expose plusieurs routes :

| Route | Méthode | Description |
|---|---|---|
| / | GET | Vérifie que l’API est active |
| /health | GET | Vérifie la disponibilité du modèle et de la base de données |
| /model-info | GET | Donne des informations sur le modèle utilisé |
| /predict | POST | Retourne une prédiction à partir d’un texte |
| /training | POST | Lance un entraînement simple depuis l’API |

La route principale est :

    POST /predict

Exemple de requête :

    {
      "text": "This product is amazing and I love it"
    }

Exemple de réponse :

    {
      "text": "This product is amazing and I love it",
      "prediction": 1,
      "label": "positive",
      "probability_negative": 0.0061,
      "probability_positive": 0.9938
    }

---

## 4. Lancer l’API

Depuis le dossier Modelisation, lancer :

    python -m uvicorn api:app --reload --port 8000

L’API est ensuite disponible à l’adresse :

    http://127.0.0.1:8000

La documentation Swagger est disponible ici :

    http://127.0.0.1:8000/docs

---

## 5. Tester manuellement l’API

Vérifier l’état de l’API :

    curl http://127.0.0.1:8000/health

Réponse attendue :

    {
      "status": "ok",
      "model_loaded": true,
      "database_available": true
    }

Faire une prédiction avec PowerShell :

    Invoke-RestMethod `
      -Method POST `
      -Uri "http://127.0.0.1:8000/predict" `
      -Headers @{"Content-Type"="application/json"} `
      -Body '{"text":"This product is amazing and I love it"}'

---

## 6. Tests automatisés

Les tests sont définis dans :

    tests/test_api.py

Ils vérifient les routes principales :

- /
- /health
- /model-info
- /predict
- validation d’une entrée vide

Pour lancer les tests :

    python -m pytest

Résultat obtenu :

    5 passed

Ces tests permettent de vérifier que l’API répond correctement avant d’intégrer le modèle dans une logique MLOps.

---

## 7. Suivi d’expériences avec MLflow

Le script d’entraînement MLOps est :

    training_mlflow.py

Il permet de :

- charger les données depuis la base SQLite ;
- préparer la cible binaire ;
- entraîner un pipeline TF-IDF + LogisticRegression ;
- calculer les métriques principales ;
- logger les paramètres dans MLflow ;
- logger les métriques dans MLflow ;
- sauvegarder le modèle localement ;
- enregistrer le modèle dans MLflow Registry ;
- comparer le nouveau modèle avec le meilleur modèle précédent.

---

## 8. Métriques suivies

Les métriques loggées dans MLflow sont :

| Métrique | Rôle |
|---|---|
| accuracy | Score global de classification |
| f1_macro | Métrique principale de comparaison |
| precision_macro | Précision moyenne entre les classes |
| recall_macro | Rappel moyen entre les classes |

La métrique principale retenue est :

    f1_macro

Ce choix est pertinent car le jeu de données est déséquilibré. La F1-macro permet de ne pas uniquement favoriser la classe majoritaire.

---

## 9. Lancer un entraînement avec MLflow

Depuis le dossier Modelisation :

    python training_mlflow.py

Le script crée un run MLflow dans l’expérience :

    trustpilot_sentiment_experiment

Le modèle est enregistré dans le Registry sous le nom :

    trustpilot_sentiment_model

---

## 10. Interface MLflow

Pour lancer l’interface MLflow :

    mlflow ui

Puis ouvrir dans le navigateur :

    http://127.0.0.1:5000

Dans l’interface, on peut consulter :

- les runs d’entraînement ;
- les paramètres ;
- les métriques ;
- les artefacts ;
- les versions du modèle dans le Registry.

---

## 11. Versioning du modèle avec MLflow Registry

À chaque entraînement, MLflow crée une nouvelle version du modèle dans le Registry.

Exemple :

    trustpilot_sentiment_model
    ├── Version 1
    └── Version 2

Le script compare automatiquement la nouvelle version avec la version actuellement marquée comme meilleure.

La meilleure version reçoit l’alias :

    best

Exemple observé :

    Version 1 -> alias best
    Version 2 -> pas d’alias

Cela signifie que la Version 2 a bien été créée, mais qu’elle n’a pas remplacé la Version 1 car elle n’était pas meilleure sur la métrique f1_macro.

---

## 12. Logique de comparaison des modèles

La logique utilisée est la suivante :

    Si aucun modèle best n’existe :
        le premier modèle est marqué comme best

    Sinon :
        comparer la f1_macro du nouveau modèle avec celle du modèle best actuel

        si la nouvelle f1_macro est meilleure :
            déplacer l’alias best vers la nouvelle version

        sinon :
            conserver l’ancien modèle best

Cette logique permet d’éviter de remplacer automatiquement un bon modèle par une version moins performante.

---

## 13. Choix techniques

### Modèle utilisé

Le modèle utilisé est un pipeline Scikit-learn :

    TF-IDF + LogisticRegression

Ce choix est adapté pour une première version robuste de classification de texte.

### API

L’API est développée avec FastAPI car ce framework permet :

- une documentation automatique avec Swagger ;
- une validation des entrées avec Pydantic ;
- une structure simple pour exposer un modèle de Machine Learning.

### MLflow

MLflow est utilisé pour :

- historiser les entraînements ;
- suivre les paramètres et les métriques ;
- versionner les modèles ;
- identifier automatiquement la meilleure version.

---

## 14. Commandes principales

Installer les dépendances :

    pip install -r requirements.txt

Lancer l’API :

    python -m uvicorn api:app --reload --port 8000

Lancer les tests :

    python -m pytest

Lancer un entraînement MLflow :

    python training_mlflow.py

Lancer l’interface MLflow :

    mlflow ui

---

## 15. Améliorations futures

Plusieurs améliorations pourront être ajoutées ensuite :

- charger directement le modèle marqué best depuis MLflow Registry dans l’API ;
- automatiser l’entraînement avec Airflow ou un cron job ;
- ajouter un pipeline CI/CD ;
- ajouter plus de tests unitaires et de tests d’intégration ;
- monitorer les prédictions en production ;
- suivre la dérive des données ;
- comparer plusieurs familles de modèles ;
- ajouter une gestion plus stricte des erreurs et des logs applicatifs.

---

## 16. Synthèse

Cette étape met en place une première chaîne MLOps fonctionnelle :

    Données SQLite
          ↓
    Préparation de la cible
          ↓
    Entraînement Scikit-learn
          ↓
    Logging MLflow
          ↓
    Registry MLflow
          ↓
    Comparaison avec le meilleur modèle précédent
          ↓
    Alias best
          ↓
    API FastAPI de prédiction
          ↓
    Tests automatisés

Le projet dispose donc d’une base exploitable pour industrialiser progressivement le modèle de prédiction d’avis Trustpilot.