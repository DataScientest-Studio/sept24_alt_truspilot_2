# Projet Trustpilot — Guide pas à pas (version simple)

Ce document explique **de bout en bout** comment répliquer et présenter le pipeline ML pour prédire la **note (1–5 étoiles)** à partir du texte d’un avis Trustpilot.

> **Public visé :** débutants + référent technique.  
> **Principe général :** nettoyage → vectorisation TF‑IDF → **rééchantillonnage** (RandomOverSampler) → modèles → métriques → optimisation.

---

## 0) Objectif & livrables
**Objectif :** classer un avis texte en 1–5 étoiles.  
**Livrables générés :**
- `data/processed/train.csv`, `data/processed/test.csv` (split stratifié)
- `models/baseline/…` (métriques, rapports, matrices de confusion, meilleur pipeline)
- `models/gridsearch_logreg/…` (meilleurs hyperparams, métriques test, pipeline optimisé)

**Métrique pivot :** `F1-macro` (chaque classe 1–5 a la **même importance**).  
**Métriques complémentaires :** `accuracy`, `balanced_accuracy`.

---

## 1) Arborescence & scripts
```
sept24_alt_truspilot_2/Modelisation/
├── data/
│   ├── trustpilot_dataset_final.csv
│   ├── trustpilot_dataset_final_clean_regex.csv          # (optionnel si exécuté)
│   └── trustpilot_dataset_final_cleaned.csv              # (optionnel si exécuté)
├── data/processed/
├── models/
│   ├── baseline/
│   └── gridsearch_logreg/
└── notebooks/
    ├── split_train_test.py
    ├── train_baselines.py
    └── gridsearch_best.py
```

---

## 2) Pré‑requis (environnement)
```bash
pip install pandas scikit-learn imbalanced-learn matplotlib joblib
# Versions conseillées (stabilité lecture/écriture modèles) :
# pip install "scikit-learn==1.4.2" "imbalanced-learn==0.12.0" "joblib>=1.3" "pandas>=2.0" matplotlib
```
> Sous Windows/PowerShell, exécuter les commandes depuis la racine du projet.

---

## 3) Données d’entrée
- **Brut** : `trustpilot_dataset_final.csv` avec colonnes `Title`, `Content`, `Rating`, …
- **Nettoyé** (optionnel) : `trustpilot_dataset_final_cleaned.csv` avec `CleanText`, `Rating`.

> Si vous partez du **brut**, les scripts reconstruisent `FullText = Title + Content`.

---

## 4) Étapes du pipeline (exécution + explication)

### 4.1 Nettoyage Regex (optionnel)
**But :** retirer URLs, emails, balises HTML, ponctuation brute, normaliser espaces.  
**Commande :**
```bash
python preprocessing_regex.py \
  --input data/trustpilot_dataset_final.csv \
  --output data/trustpilot_dataset_final_clean_regex.csv
```
**Sortie :** ajoute `FullText` et `CleanText_regex` (utile si vous voulez inspecter le texte nettoyé).

### 4.2 Prétraitement avancé (optionnel)
**But :** produire `CleanText` final (minuscule → stopwords anglais → lemmatisation).  
**Commande :**
```bash
python preprocessing_advanced.py \
  --input data/trustpilot_dataset_final_clean_regex.csv \
  --output data/trustpilot_dataset_final_cleaned.csv
```
**Sortie :** `CleanText`, `Rating`.

> **Astuce** : Pour une première passe “simple et présentable”, vous pouvez **sauter** 4.1 et 4.2 si vous possédez déjà un CSV avec `CleanText` + `Rating`.

### 4.3 Split train/test (stratifié)
**But :** séparer en train/test en conservant la proportion des 5 classes.  
**Commande :**
```bash
python notebooks/split_train_test.py \
  --input data/trustpilot_dataset_final_cleaned.csv \
  --outdir data/processed
```
**Sorties :**
- `data/processed/train.csv` (colonnes `CleanText`, `Rating`)
- `data/processed/test.csv` (colonnes `CleanText`, `Rating`)

**Pourquoi stratifier ?**  Pour que le test contienne des 1★…5★ dans des proportions comparables au train.

### 4.4 Entraînement baselines + rééchantillonnage
**But :** comparer 3 modèles simples avec un pipeline **TF‑IDF → RandomOverSampler → Modèle**.

**Commande :**
```bash
python notebooks/train_baselines.py \
  --train data/processed/train.csv \
  --test data/processed/test.csv \
  --outdir models/baseline
```
**Ce que fait le script :**
1) **TF‑IDF** (1–2‑gram, `max_features=5000`) : transforme le texte en vecteurs numériques.  
2) **RandomOverSampler** : **équilibre** le train en dupliquant aléatoirement les classes minoritaires (p. ex. 2★, 3★).  
3) Modèles testés : `LogisticRegression`, `LinearSVC`, `RandomForestClassifier`.  
4) **Métriques** (sur test **non rééchantillonné**) : `accuracy`, `balanced_accuracy`, `f1_macro`.

**Sorties :**
- `models/baseline/baseline_metrics.csv` (comparatif)
- `models/baseline/<model>/classification_report.txt`
- `models/baseline/<model>/confusion_matrix.png`
- `models/baseline/best_baseline.joblib` (pipeline complet le mieux classé en **F1‑macro**)

**À expliquer simplement :**
- Le **rééchantillonnage** est **intégré au pipeline** et n’agit **que pendant l’entraînement** (jamais sur le test).  
- On met en avant **F1‑macro** pour donner le **même poids** aux 5 classes (évite que 5★ domine tout).

### 4.5 Optimisation (GridSearchCV) — LogReg
**But :** chercher automatiquement des hyperparamètres qui maximisent **F1‑macro**.

**Commande :**
```bash
python notebooks/gridsearch_best.py \
  --train data/processed/train.csv \
  --test data/processed/test.csv \
  --outdir models/gridsearch_logreg
```
**Grille (exemple) :**
- TF‑IDF : `max_features` (5k, 10k, 20k), `ngram_range` ((1,1), (1,2), (1,3)), `min_df` (1,2,3)  
- LogReg : `C` (0.5, 1, 2, 4), `class_weight` (None, balanced)

**Sorties :**
- `best_params.json` — hyperparams sélectionnés
- `test_metrics.json` — scores finaux sur test
- `classification_report.txt` — détails par classe
- `best_model.joblib` — pipeline complet optimisé

**Interprétation typique :**
- **Tri‑grammes** (1–3) et vocabulaire élargi (jusqu’à 20k) captent davantage d’expressions utiles.
- `C = 0.5` → régularisation plus forte (meilleure généralisation).  
- `class_weight=None` souvent retenu si l’**oversampling** suffit à équilibrer l’apprentissage.

---

## 5) Inférence : utiliser le modèle `.joblib`
Un fichier `.joblib` est un **objet Python** (pipeline). On le **charge dans Python** :

```python
import imblearn  # important si le pipeline utilise imblearn.Pipeline
import joblib

pipe = joblib.load("models/gridsearch_logreg/best_model.joblib")  # ou models/baseline/best_baseline.joblib

texts = [
    "Amazing visit, friendly staff and quick support. Highly recommend!",
    "Terrible experience. Item never arrived and customer service was rude."
]

pred = pipe.predict(texts)  # → array d’étiquettes [1..5]
print(pred)

# Si le modèle le permet (LogReg / RF) :
try:
    proba = pipe.predict_proba(texts)  # probas par classe
    print(proba)
except Exception:
    pass
```

**Erreurs fréquentes :**
- `ModuleNotFoundError: imblearn…` → `pip install imbalanced-learn` + `import imblearn` **avant** le `load`.
- Différences de versions scikit‑learn → alignez les versions (voir §2 Pré‑requis).

---

## 6) Comment lire les métriques & matrices de confusion
- **Accuracy** : part des prédictions correctes **toutes classes confondues** (peut être trompeuse si 5★ est majoritaire).
- **Balanced Accuracy** : moyenne des recalls par classe (corrige partiellement le déséquilibre).
- **F1‑macro** : moyenne des F1 par classe (**métrique pivot**).  

**Matrices de confusion** (PNG dans `models/.../<model>/`) : chaque ligne = vraie classe, chaque colonne = classe prédite.  
Regardez surtout :
- Confusions **4★ ↔ 5★** (classique)  
- Confusions **2★ ↔ 1★/3★** (frontières “mécontent” vs “mitigé”)

---

## 7) Script‑by‑script — résumé prêt à dire
- **`split_train_test.py`** : crée `train.csv`/`test.csv` (stratifié), en partant soit de `CleanText` soit de `Title+Content`.
- **`train_baselines.py`** : TF‑IDF → **RandomOverSampler** → 3 modèles ; compare `accuracy`, `balanced_accuracy`, **`F1‑macro`** ; génère rapports + matrices.
- **`gridsearch_best.py`** : optimise la LogReg (grille sur TF‑IDF et C) en **F1‑macro** ; sauvegarde meilleur pipeline + métriques test.

---

## 8) Pitch oral (30–45 sec)
> “On prédit la note 1–5 d’un avis. On nettoie le texte, on l’encode en TF‑IDF, on **rééquilibre** les classes rares avec un sur‑échantillonnage aléatoire, puis on compare 3 modèles.  
> On choisit la **F1‑macro** pour donner le même poids aux 5 classes.  
> La **régression logistique** performe le mieux en F1‑macro (~0,61) avec tri‑grammes et un vocabulaire élargi ; l’accuracy se stabilise autour de ~0,79.  
> On produit métriques, matrices de confusion et un pipeline **.joblib** réutilisable.”

---

## 9) Dépannage rapide
- **IndentationError / SyntaxError** : recoller le code “propre” (l’IDE a pu couper une ligne).  
- **`KeyError: 'CleanText'`** : repartir du CSV brut en entrée du `split`, ou rejouer le preprocessing avancé.  
- **`ModuleNotFoundError: imblearn`** : `pip install imbalanced-learn` + `import imblearn` avant `joblib.load`.  
- **Versions scikit‑learn incompatibles** : installer les versions conseillées (cf. §2).  
- **Encodage** : ajouter `encoding="utf-8"` dans `pd.read_csv` si besoin.

---

## 10) (Annexe — pistes d’amélioration, facultatif)
- **SVM (LinearSVC) avec GridSearch** : parfois +1–2 points F1‑macro.
- **Features simples** : longueur du texte, `!` et `?`, sentiment (VADER/TextBlob) → concaténés à TF‑IDF via `FeatureUnion`.
- **Class weights** : à tester **sans** oversampling (ou bien l’un ou l’autre, pas les deux en même temps).

---

**Fin du guide.**  
Besoin d’un **README.md téléchargeable** ou de **slides** prêtes pour la soutenance ? Dites‑le et je l’exporte au bon format.

