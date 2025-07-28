import pandas as pd
import re
import emoji
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from symspellpy import SymSpell, Verbosity
import string

# 1. Hygiène générale du jeu
def retirer_doublons_et_lignes_vides(df, champs_critiques):
    df = df.drop_duplicates()
    df = df.dropna(subset=champs_critiques)
    return df

# 2. Nettoyage HTML / URLs / emojis
def nettoyer_html_url_emoji(texte):
    # Supprimer balises <br> et <br/> et liens
    texte = re.sub(r'<br\s*/?>', ' ', texte)
    texte = re.sub(r'http\S+|www\.\S+', '', texte)
    # Conversion emojis en texte
    texte = emoji.demojize(texte, delimiters=(" ", " "))
    return texte

def appliquer_nettoyage_html_url_emoji(df, colonne):
    df[colonne] = df[colonne].astype(str).apply(nettoyer_html_url_emoji)
    return df

# 3. Normalisation orthographique
def normaliser_texte(texte, sym_spell=None):
    # Lower-case
    texte = texte.lower()
    # Décontraction
    contractions = {"didn't": "did not", "can't": "cannot", "won't": "will not", "i'm": "i am", "it's": "it is", "you're": "you are", "they're": "they are", "we're": "we are", "isn't": "is not", "aren't": "are not", "wasn't": "was not", "weren't": "were not", "hasn't": "has not", "haven't": "have not", "hadn't": "had not", "doesn't": "does not", "don't": "do not", "didn't": "did not", "wouldn't": "would not", "shouldn't": "should not", "couldn't": "could not", "mustn't": "must not", "let's": "let us", "that's": "that is", "who's": "who is", "what's": "what is", "here's": "here is", "there's": "there is", "where's": "where is", "when's": "when is", "why's": "why is", "how's": "how is"}
    for c, full in contractions.items():
        texte = re.sub(r'\\b' + re.escape(c) + r'\\b', full, texte)
    # Correction orthographique légère
    if sym_spell:
        tokens = texte.split()
        corrected = []
        for token in tokens:
            suggestions = sym_spell.lookup(token, Verbosity.CLOSEST, max_edit_distance=2)
            if suggestions:
                corrected.append(suggestions[0].term)
            else:
                corrected.append(token)
        texte = ' '.join(corrected)
    return texte

def appliquer_normalisation(df, colonne, sym_spell=None):
    df[colonne] = df[colonne].astype(str).apply(lambda x: normaliser_texte(x, sym_spell))
    return df

# 4. Features structurés
def extraire_features_structures(df, colonne_texte, colonne_date_commande=None, colonne_date_avis=None):
    df['longueur_char'] = df[colonne_texte].astype(str).apply(len)
    df['longueur_phrase'] = df[colonne_texte].astype(str).apply(lambda x: len(re.split(r'[.!?]', x)))
    df['nb_majuscules'] = df[colonne_texte].astype(str).apply(lambda x: sum(1 for c in x if c.isupper()))
    df['nb_exclamations'] = df[colonne_texte].astype(str).apply(lambda x: x.count('!'))
    if colonne_date_commande and colonne_date_avis:
        df['delai_avis'] = pd.to_datetime(df[colonne_date_avis]) - pd.to_datetime(df[colonne_date_commande])
        df['delai_avis'] = df['delai_avis'].dt.days
    return df

# 5. Équilibrage des classes
def equilibrer_classes(df, colonne_label, colonne_texte, sous_echantillon_5=True, sur_echantillon_1=True):
    X = df[colonne_texte]
    y = df[colonne_label]
    vectorizer = TfidfVectorizer(max_features=1000)
    X_vect = vectorizer.fit_transform(X)
    if sous_echantillon_5:
        rus = RandomUnderSampler(sampling_strategy={5: int((y==5).sum()*0.2)}, random_state=42)
        X_vect, y = rus.fit_resample(X_vect, y)
    if sur_echantillon_1:
        smote = SMOTE(sampling_strategy={1: int((y==1).sum()*2)}, random_state=42)
        X_vect, y = smote.fit_resample(X_vect, y)
    return X_vect, y

def enlever_lignes_valeurs_manquantes(df, chemin_sortie):
    """
    Supprime toutes les lignes contenant au moins une valeur manquante et sauvegarde le résultat dans un nouveau fichier CSV.
    """
    df_sans_na = df.dropna()
    df_sans_na.to_csv(chemin_sortie, index=False)
    print(f'Fichier sans valeurs manquantes sauvegardé : {chemin_sortie}')
    return df_sans_na

# Chargement et test de chaque fonction
def main():
    # Exemple de chargement de données
    df = pd.read_csv('/Users/martywong/Downloads/Truspilot-Modelisation/trustpilot_ringconn_v3.csv')
    champs_critiques = ['Content', 'Rating']
    colonne_texte = 'Content'
    colonne_label = 'Rating'
    colonne_date_commande = 'date_commande' if 'date_commande' in df.columns else None
    colonne_date_avis = 'date_avis' if 'date_avis' in df.columns else None

    print('1. Hygiène générale')
    df = retirer_doublons_et_lignes_vides(df, champs_critiques)
    print(df.head())

    print('2. Nettoyage HTML/URL/emoji')
    df = appliquer_nettoyage_html_url_emoji(df, colonne_texte)
    print(df.head())

    print('3. Normalisation orthographique')
    sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym_spell.load_dictionary('frequency_dictionary_en_50k.txt', term_index=0, count_index=1)
    df = appliquer_normalisation(df, colonne_texte, sym_spell)
    print(df.head())

    print('4. Features structurés')
    df = extraire_features_structures(df, colonne_texte, colonne_date_commande, colonne_date_avis)
    print(df.head())

    # Sauvegarde du résultat final
    df.to_csv('/Users/martywong/Downloads/Truspilot-Modelisation/trustpilot_preprocessed.csv', index=False)
    print('Fichier trustpilot_preprocessed.csv sauvegardé.')

    # Suppression des lignes avec valeurs manquantes et sauvegarde
    enlever_lignes_valeurs_manquantes(
        df,
        '/Users/martywong/Downloads/Truspilot-Modelisation/trustpilot_preprocessed_noNA.csv'
    )

    print('5. Équilibrage des classes')
    X_vect, y = equilibrer_classes(df, colonne_label, colonne_texte)
    print('X_vect shape:', X_vect.shape)
    print('y shape:', y.shape)

if __name__ == "__main__":
    main() 