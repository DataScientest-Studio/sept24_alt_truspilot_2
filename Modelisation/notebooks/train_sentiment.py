# train_sentiment_gb_tfidf.py
import argparse
import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

os.makedirs("Modelisation/models", exist_ok=True)

def load_clean_dataset(path_hint: str):
    """
    Charge le dataset nettoyé.
    """
    p = Path(path_hint)
    if p.exists():
        return pd.read_csv(p)
    # fallback fréquents
    candidates = [
        "trustpilot_dataset_final_features.csv"
    ]
    for c in candidates:
        if Path(c).exists():
            return pd.read_csv(c) if c.endswith(".csv") else pd.read_csv(c, sep="\t")
    raise FileNotFoundError(
        f"Fichier introuvable. Passé: {path_hint}. "
        f"Essais: {', '.join(candidates)}"
    )

def binarize_rating(df: pd.DataFrame, text_col="CleanText", rating_col="Rating"):
    """
    Construit une cible binaire:
      - 4,5 -> 1 (positif)
      - 1,2 -> 0 (négatif)
      - 3 -> neutre (éliminé du training)
    Ne garde que les lignes avec du texte non vide.
    """
    if text_col not in df.columns or rating_col not in df.columns:
        raise ValueError(f"Colonnes requises manquantes: '{text_col}' et/ou '{rating_col}'.")

    dfx = df[[text_col, rating_col]].copy()
    # rating peut être string -> numeric
    dfx[rating_col] = pd.to_numeric(dfx[rating_col], errors="coerce")

    # map binaire
    def map_label(r):
        if pd.isna(r): return np.nan
        if r >= 4: return 1
        if r <= 2: return 0
        return np.nan  # 3 -> neutre

    dfx["Sentiment"] = dfx[rating_col].apply(map_label)
    # drop neutres / NaN / textes vides
    dfx[text_col] = dfx[text_col].astype(str).str.strip()
    dfx = dfx[(dfx["Sentiment"].notna()) & (dfx[text_col] != "")]
    dfx["Sentiment"] = dfx["Sentiment"].astype(int)
    return dfx.rename(columns={text_col: "Text"})

def main(input_path, model_out, vectorizer_out, test_size=0.2, random_state=30):
    # 1) Charger le dataset prétraité
    df = load_clean_dataset(input_path)

    # 2) Binariser la cible et préparer X/y
    dfx = binarize_rating(df, text_col="CleanText", rating_col="Rating")
    X, y = dfx["Text"].values, dfx["Sentiment"].values

    # Petit aperçu des classes
    class_counts = pd.Series(y).value_counts().sort_index()
    print("Répartition des classes (0=neg, 1=pos):")
    print(class_counts.to_string(), "\n")

    # 3) Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4) TF-IDF (paramètres par défaut, comme dans le cours)
    vec = TfidfVectorizer()
    X_train_tfidf = vec.fit_transform(X_train)
    X_test_tfidf = vec.transform(X_test)

    # 5) Gradient Boosting (hyperparams du cours)
    clf = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=1.0,
        max_depth=1,
        random_state=0
    )
    clf.fit(X_train_tfidf, y_train)

    # 6) Évaluation
    y_pred = clf.predict(X_test_tfidf)
    print("Classification report (TF-IDF + GradientBoosting):")
    print(classification_report(y_test, y_pred))
    print("Matrice de confusion:")
    print(confusion_matrix(y_test, y_pred))

    # 7) Exemple de prédiction (tu peux adapter)
    comments = [
        "The ring is fantastic and the sleep analysis is spot on.",
        "Battery died after one day and support ignored my emails.",
        "Works as expected.",
        "Shipping was slow but the device quality is impressive.",
        "Terrible experience, I want a refund."
    ]
    tok = vec.transform(comments)
    preds = clf.predict(tok)
    print("\nExemples de prédiction :")
    for c, p in zip(comments, preds):
        lab = "positif (1)" if p == 1 else "négatif (0)"
        print(f"- {c} -> {lab}")

    # 8) Sauvegarde du modèle et du vectorizer
    joblib.dump(clf, model_out)
    joblib.dump(vec, vectorizer_out)
    print(f"\n✅ Modèle sauvegardé: {model_out}")
    print(f"✅ Vectorizer sauvegardé: {vectorizer_out}")

if __name__ == "__main__":
       # Valeurs par défaut
    input_file = "trustpilot_dataset_final_features.csv"
    model_path = "Modelisation/models/gb_tfidf_model.joblib"
    vectorizer_path = "Modelisation/models/tfidf_vectorizer.joblib"
    test_size = 0.2
    random_state = 30

    # Lancement direct
    main(input_file, model_path, vectorizer_path, test_size, random_state)
