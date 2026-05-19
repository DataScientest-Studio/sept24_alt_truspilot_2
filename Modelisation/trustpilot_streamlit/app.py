import streamlit as st
import joblib
import numpy as np
from pathlib import Path

# --------------------------------------------------
# CONFIG GLOBALE
# --------------------------------------------------
st.set_page_config(
    page_title="Trustpilot – Analyse & Prédiction de Sentiment",
    page_icon="📊",
    layout="wide"
)

FIG_DIR = Path("figures")
MODEL_PATH = "binary_logreg.joblib"

# --------------------------------------------------
# CHARGEMENT DU MODÈLE
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("📊 Analyse & Prédiction de Sentiment – Avis Trustpilot")
st.markdown(
    """
    Projet de **Data Science – NLP**  
    Objectif : **prédire automatiquement le sentiment d’un avis client**
    (positif / négatif) à partir de son texte.
    """
)

# --------------------------------------------------
# ONGLETS
# --------------------------------------------------
tab_context, tab_demarche, tab_explo, tab_comparaisons, tab_model, tab_limites, tab_demo = st.tabs([
    "🏠 Contexte & objectif",
    "🧭 Démarche méthodologique",
    "📊 Exploration des données",
    "🧪 Comparaisons & arbitrages",   
    "🤖 Modélisation finale",
    "⚠️ Limites & perspectives",
    "🔮 Démo – Prédiction"
])


# ==================================================
# 🏠 CONTEXTE
# ==================================================
with tab_context:
    st.header("🎯 Contexte & Objectifs")

    st.markdown(
        """
        Les plateformes d’avis clients comme **Trustpilot** contiennent une grande quantité
        de feedback textuel difficile à analyser manuellement.

        **Objectif du projet :**
        - Exploiter ces avis via le **traitement automatique du langage (NLP)**
        - Construire un **modèle de classification binaire**
        - Fournir un outil de **prédiction instantanée hintée métier**
        """
    )

    st.info(
        "⚙️ Cas d’usage métier : détection rapide d’insatisfaction client, "
        "priorisation du support, analyse de réputation."
    )
# ==================================================
# 🧠 DÉMARCHE DU PROJET (ONGLET CLÉ)
# ==================================================
with tab_demarche:
    st.header("🧠 Démarche méthodologique")

    st.markdown("""
    ### Objectif initial
    Le projet visait à analyser automatiquement le sentiment des avis clients Trustpilot
    à partir de leur contenu textuel.

    ### Hypothèse de départ
    Le problème a d’abord été formulé comme une **classification multiclasse**
    (notes de 1 à 5 étoiles), correspondant à la structure brute des données.

    ### Difficultés rencontrées
    L’analyse des premières matrices de confusion a montré une **forte ambiguïté**
    entre les classes intermédiaires (3–4), rendant l’interprétation métier difficile
    et les performances instables.

    ### Décision méthodologique
    Le problème a été reformulé en **classification binaire** :
    - 0 : avis négatif (1–2)
    - 1 : avis positif (3–4–5)

    Ce choix permet d’aligner le modèle avec un besoin métier réel
    (détection d’insatisfaction) et d’améliorer la robustesse globale.

    ### Choix assumés
    Le projet a volontairement privilégié un **pipeline simple, interprétable
    et reproductible**, plutôt qu’une complexité algorithmique excessive.
    """)

    # ==================================================
# 📈 EXPLORATION
# ==================================================
with tab_explo:
    st.header("📈 Exploration des données")

    col1, col2 = st.columns(2)

    with col1:
        st.image(FIG_DIR / "distribution_longueur_avis.png", use_container_width=True)
        st.caption(
            "La majorité des avis sont courts, avec une longue traîne d’avis très détaillés. "
            "Le TF-IDF est bien adapté à cette variabilité de longueur."
        )

    with col2:
        st.image(FIG_DIR / "nombre_avis_par_mois.png", use_container_width=True)
        st.caption(
            "Le volume d’avis varie fortement dans le temps, sans saisonnalité stricte "
            "imposant une contrainte temporelle au modèle."
        )

    st.divider()

    col3, col4, col5 = st.columns(3)

    with col3:
        st.image(FIG_DIR / "wc_all.png", use_container_width=True)
        st.caption(
            "Les termes dominants concernent le produit, l’expérience et le suivi client."
        )

    with col4:
        st.image(FIG_DIR / "wc_negative.png", use_container_width=True)
        st.caption(
            "Les avis négatifs font ressortir des mots liés aux problèmes, délais et retours."
        )

    with col5:
        st.image(FIG_DIR / "wc_positive.png", use_container_width=True)
        st.caption(
            "Les avis positifs sont marqués par un vocabulaire émotionnel et affirmatif."
        )

    st.divider()

    col6, col7, col8 = st.columns(3)

    with col6:
        st.image(FIG_DIR / "hist_all_top20.png", use_container_width=True)
        st.caption(
            "Les mots fréquents sont génériques, ce qui justifie l’usage de bigrams."
        )

    with col7:
        st.image(FIG_DIR / "hist_neg_top20.png", use_container_width=True)
        st.caption(
            "Les avis négatifs présentent un vocabulaire plus spécifique et discriminant."
        )

    with col8:
        st.image(FIG_DIR / "hist_pos_top20.png", use_container_width=True)
        st.caption(
            "Les avis positifs utilisent un lexique plus répétitif et homogène."
        )
# ==================================================
# 🧪 COMPARAISONS & ARBITRAGES
# ==================================================
with tab_comparaisons:
    st.header("🧪 Comparaisons et arbitrages méthodologiques")

    st.markdown("""
    Le projet a été mené de manière **itérative**, avec plusieurs formulations du problème
    testées successivement.  
    Chaque choix de modélisation a été **guidé par l’analyse des erreurs**, la stabilité
    des résultats et leur **cohérence métier**.
    """)

    st.subheader("🔍 Approches testées")

    st.table({
        "Approche testée": [
            "Classification multi-classe (1 à 5 étoiles)",
            "Classification binaire (avis négatif / positif)",
            "Classification binaire + features de sentiment"
        ],
        "Objectif initial": [
            "Prédire précisément la note attribuée par l’utilisateur",
            "Simplifier la tâche et réduire l’ambiguïté sémantique",
            "Tester un enrichissement sémantique du texte"
        ],
        "Constats principaux": [
            "Confusions importantes entre classes intermédiaires (2–3–4)",
            "Séparation plus nette entre avis satisfaits et insatisfaits",
            "Amélioration marginale au prix d’un pipeline plus complexe"
        ],
        "Décision": [
            "Approche abandonnée",
            "Approche retenue",
            "Non retenue"
        ]
    })

    st.markdown("""
    Cette comparaison met en évidence un point clé du projet :  
    **augmenter la complexité du modèle n’implique pas nécessairement de meilleures performances**.
    Le choix final repose sur un compromis assumé entre **performance**, **robustesse**
    et **interprétabilité**.
    """)

    st.subheader("📊 Justification du pivot multi-classe → binaire")

    st.markdown("""
    Les matrices de confusion ci-dessous illustrent visuellement ce choix.
    Elles montrent pourquoi la classification binaire s’est révélée plus adaptée
    au contexte du projet.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.caption("Matrice de confusion — classification multi-classe (normalisée)")
        st.image(
            "figures/confusion_matrix_multiclass_normalized.png",
            use_container_width=True
        )
        st.markdown("""
        👉 Les classes intermédiaires (2, 3 et 4 étoiles) sont fortement confondues.  
        Cela s’explique par la proximité sémantique des avis mitigés, difficiles à
        distinguer uniquement par le texte.
        """)

    with col2:
        st.caption("Matrice de confusion — classification binaire")
        st.image(
            "figures/confusion_matrix_binary_opt.png",
            use_container_width=True
        )
        st.markdown("""
        👉 La séparation entre avis négatifs et positifs est beaucoup plus nette.  
        Cette formulation permet d’obtenir un modèle **plus stable**, **plus lisible**
        et **directement exploitable** d’un point de vue métier.
        """)




# ==================================================
# 🧠 MODÉLISATION
# ==================================================
with tab_model:
    st.header("🧠 Modélisation & Évaluation")

    st.markdown("""
    ### Logique de choix du modèle

    Plusieurs approches ont été envisagées au cours du projet :
    - formulation multiclasse vs binaire,
    - modèles baseline vs modèle final,
    - tests avec et sans features de sentiment.

    Ces comparaisons ont montré que l’augmentation de la complexité
    n’apportait pas de gain significatif et nuisait parfois à la stabilité.

    Le modèle final correspond donc à un **compromis assumé**
    entre performance, interprétabilité et robustesse.
    """)


    col9, col10 = st.columns(2)

    with col9:
        st.image(FIG_DIR / "confusion_matrix_binary_opt.png", use_container_width=True)
        st.caption(
            "Bonne séparation entre avis positifs et négatifs, avec un compromis "
            "précision / rappel adapté à un contexte métier."
        )

    with col10:
        st.image(FIG_DIR / "roc.png", use_container_width=True)
        st.caption(
            "La courbe ROC indique une forte capacité de discrimination du modèle."
        )

    col_pr_left, col_pr_center, col_pr_right = st.columns([1, 2, 1])

    with col_pr_center:
        st.image(FIG_DIR / "pr.png", use_container_width=True)
        st.caption(
            "La courbe Precision-Recall confirme de bonnes performances malgré le déséquilibre "
            "des classes, justifiant l’usage du F1-score."
        )

# ==================================================
# ⚠️ LIMITES & PISTES D’AMÉLIORATION
# ==================================================
with tab_limites:
    st.header("⚠️ Limites et pistes d'amélioration")

    st.markdown("""
    ### Limites actuelles du projet

    - Le modèle repose sur TF-IDF : il capture le vocabulaire,
      mais pas le sens profond du texte.
    - Les avis courts ou peu expressifs génèrent une incertitude élevée.
    - Le modèle est sensible à la langue (anglais dominant).
    - Les performances dépendent fortement du domaine d’entraînement.

    ### Améliorations possibles

    - Utilisation d’embeddings sémantiques (Word2Vec, BERT).
    - Gestion multilingue.
    - Données supplémentaires ou annotations métier.
    """)

    st.markdown("""
    Ces limites sont connues, assumées et constituent
    des pistes d’amélioration claires du projet.
    """)

# ==================================================
# 🔮 DÉMO
# ==================================================
with tab_demo:
    st.header("🔮 Démonstration – Prédiction en direct")

    st.markdown(
        "Entrez un **avis client** pour prédire automatiquement son sentiment "
        "(positif / négatif)."
    )

    user_text = st.text_area(
        "✍️ Avis client",
        value="Great product, very happy with the experience",
        height=120
    )

    if st.button("✨ Prédire le sentiment"):
        if user_text.strip() == "":
            st.warning("Veuillez entrer un texte.")
        else:
            proba = model.predict_proba([user_text])[0, 1]
            pred = int(proba >= 0.5)

            if pred == 1:
                st.success("✅ Avis POSITIF")
            else:
                st.error("❌ Avis NÉGATIF")

            st.info(f"📊 Probabilité de positivité : **{proba*100:.1f}%**")

    st.caption(
        "⚠️ Modèle entraîné majoritairement sur des avis en langue anglaise. "
        "Aucun ré-entraînement n’est effectué dans l’application."
    )
