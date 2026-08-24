from typing import Optional, Dict, Any
import random
import time

import requests
import streamlit as st


# ==================================================
# CONFIGURATION GLOBALE
# ==================================================

st.set_page_config(
    page_title="Trustpilot Sentiment – Interface MLOps",
    page_icon="🧭",
    layout="wide"
)

DEFAULT_API_URL = "http://127.0.0.1:8001"
DEFAULT_API_KEY = "trustpilot-secret-key"
DEFAULT_GRAFANA_URL = "http://127.0.0.1:3000"
DEFAULT_PROMETHEUS_URL = "http://127.0.0.1:9090"
DEFAULT_MLFLOW_URL = "http://127.0.0.1:5001"
DEFAULT_AIRFLOW_URL = "http://127.0.0.1:8080"


# ==================================================
# FONCTIONS UTILITAIRES
# ==================================================

def call_api_predict(api_url: str, api_key: str, text: str) -> Dict[str, Any]:
    """
    Appelle l'endpoint sécurisé /predict de l'API FastAPI.
    """
    endpoint = f"{api_url.rstrip('/')}/predict"

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key
    }

    response = requests.post(
        endpoint,
        headers=headers,
        json={"text": text},
        timeout=10
    )

    response.raise_for_status()
    return response.json()


def call_api_health(api_url: str) -> Optional[Dict[str, Any]]:
    """
    Appelle l'endpoint public /health de l'API.
    """
    try:
        endpoint = f"{api_url.rstrip('/')}/health"
        response = requests.get(endpoint, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def query_prometheus(prometheus_url: str, query: str) -> Optional[float]:
    """
    Exécute une requête Prometheus instantanée.
    Retourne la première valeur numérique trouvée.
    """
    try:
        endpoint = f"{prometheus_url.rstrip('/')}/api/v1/query"

        response = requests.get(
            endpoint,
            params={"query": query},
            timeout=5
        )

        response.raise_for_status()
        data = response.json()

        results = data.get("data", {}).get("result", [])

        if not results:
            return None

        value = results[0].get("value", [None, None])[1]

        if value is None:
            return None

        return float(value)

    except Exception:
        return None


def safe_percent(value: Optional[float]) -> str:
    """
    Affiche un ratio Prometheus sous forme de pourcentage.
    """
    if value is None:
        return "N/A"

    return f"{value * 100:.1f} %"


def probability_bar(label: str, value: Optional[float], color: str) -> None:
    """
    Affiche une barre de probabilité en HTML simple.
    On évite st.progress pour éviter les erreurs JS Streamlit.
    """
    if value is None:
        st.write(f"{label} : N/A")
        return

    percent = max(0, min(100, value * 100))

    st.markdown(
        f"""
        <div style="margin-bottom: 1rem;">
            <p style="margin-bottom: 0.3rem;">
                <strong>{label}</strong> : {percent:.1f} %
            </p>
            <div style="
                background-color: #2b2b2b;
                border-radius: 8px;
                height: 22px;
                width: 100%;
                overflow: hidden;
            ">
                <div style="
                    background-color: {color};
                    width: {percent:.1f}%;
                    height: 22px;
                    border-radius: 8px;
                "></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def generate_demo_predictions(
    api_url: str,
    api_key: str,
    count: int = 50,
    mode: str = "balanced"
) -> Dict[str, int]:
    """
    Génère des appels de démonstration vers l'API /predict
    afin d'alimenter les métriques Prometheus/Grafana.
    """

    positive_texts = [
        "This product is amazing. Delivery was fast and customer support was very helpful.",
        "Excellent service, I am very satisfied with my order.",
        "Great experience, fast delivery and good quality.",
        "I recommend this company, everything was perfect.",
        "Very happy with my purchase.",
        "The product works perfectly and the delivery was fast.",
        "Amazing customer support and great quality.",
        "The experience was smooth and the team was helpful."
    ]

    negative_texts = [
        "This is terrible. The product arrived broken and customer service never answered.",
        "Bad experience, the product arrived damaged.",
        "Worst service ever, I want a refund.",
        "I am not satisfied with this order.",
        "The delivery was late and the support was useless.",
        "Very poor quality, I will never buy again.",
        "Customer service was awful.",
        "The product does not work at all."
    ]

    mixed_texts = [
        "The product is okay, but delivery was late and the support experience could be better.",
        "The service was acceptable, but I expected more.",
        "Not bad overall, but some details were disappointing.",
        "The quality is fine, but the delivery process was frustrating."
    ]

    if mode == "Majoritairement positif":
        demo_texts = positive_texts * 4 + negative_texts + mixed_texts
    elif mode == "Majoritairement négatif":
        demo_texts = negative_texts * 4 + positive_texts + mixed_texts
    else:
        demo_texts = positive_texts + negative_texts + mixed_texts

    results = {
        "total": 0,
        "positive": 0,
        "negative": 0,
        "errors": 0,
    }

    for _ in range(count):
        text = random.choice(demo_texts)

        try:
            response = call_api_predict(api_url, api_key, text)
            label = response.get("label")

            results["total"] += 1

            if label == "positive":
                results["positive"] += 1
            elif label == "negative":
                results["negative"] += 1

            time.sleep(0.10)

        except Exception:
            results["errors"] += 1

    return results


# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Configuration locale")

api_url = st.sidebar.text_input(
    "URL API FastAPI",
    value=DEFAULT_API_URL
)

api_key = st.sidebar.text_input(
    "Clé API utilisée pour appeler /predict",
    value=DEFAULT_API_KEY,
    type="password"
)

grafana_url = st.sidebar.text_input(
    "URL Grafana",
    value=DEFAULT_GRAFANA_URL
)

prometheus_url = st.sidebar.text_input(
    "URL Prometheus",
    value=DEFAULT_PROMETHEUS_URL
)

mlflow_url = st.sidebar.text_input(
    "URL MLflow",
    value=DEFAULT_MLFLOW_URL
)

airflow_url = st.sidebar.text_input(
    "URL Airflow",
    value=DEFAULT_AIRFLOW_URL
)

st.sidebar.divider()

st.sidebar.markdown(
    """
    **Services attendus :**

    - FastAPI : `8001`
    - MLflow : `5001`
    - Airflow : `8080`
    - Prometheus : `9090`
    - Grafana : `3000`
    """
)


# ==================================================
# HEADER
# ==================================================

st.title("🧭 Trustpilot Sentiment – Interface utilisateur MLOps")

st.markdown(
    """
    Cette interface permet à un utilisateur métier de tester le modèle de sentiment
    sans manipuler directement le code Python.

    L’application appelle une **API FastAPI sécurisée**, affiche le résultat de prédiction
    et donne accès au monitoring en direct via **Prometheus** et **Grafana**.
    """
)


# ==================================================
# ONGLETS
# ==================================================

tab_home, tab_predict, tab_monitoring, tab_pipeline, tab_docs = st.tabs([
    "🏠 Accueil",
    "🔮 Prédiction",
    "📈 Monitoring live",
    "🧱 Pipeline MLOps",
    "📚 Documentation & limites"
])


# ==================================================
# ONGLET ACCUEIL
# ==================================================

with tab_home:
    st.header("🎯 Objectif de l’application")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
            Le projet vise à classifier automatiquement des avis clients Trustpilot
            en deux catégories :

            - **Avis positif**
            - **Avis négatif / insatisfait**

            L’objectif n’est pas seulement d’avoir un modèle performant, mais de montrer
            comment ce modèle peut être **mis à disposition**, **sécurisé**, **monitoré**
            et **maintenu** dans une logique MLOps.
            """
        )

        st.info(
            "Cas d’usage métier : aider une équipe support, marketing ou réputation client "
            "à repérer rapidement les avis négatifs et suivre l’évolution du sentiment client."
        )

    with col2:
        st.metric("Type de modèle", "NLP")
        st.metric("Sortie", "Positif / Négatif")
        st.metric("Interface", "Streamlit + API")

    st.divider()

    st.subheader("🧩 Composants utilisés")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("### 🚀 FastAPI")
        st.write("Expose le modèle via une API sécurisée.")

    with c2:
        st.markdown("### 📊 MLflow")
        st.write("Suit les entraînements et versionne les modèles.")

    with c3:
        st.markdown("### 🕒 Airflow")
        st.write("Orchestre le pipeline d’entraînement.")

    with c4:
        st.markdown("### 📈 Grafana")
        st.write("Affiche le monitoring et le drift proxy.")


# ==================================================
# ONGLET PRÉDICTION
# ==================================================

with tab_predict:
    st.header("🔮 Prédire le sentiment d’un avis client")

    st.markdown(
        """
        L’utilisateur saisit un avis client.  
        L’application Streamlit envoie ensuite la requête à l’API FastAPI sécurisée.
        """
    )

    example_choice = st.selectbox(
        "Exemples rapides",
        [
            "Avis positif",
            "Avis négatif",
            "Avis mitigé",
            "Texte personnalisé"
        ]
    )

    examples = {
        "Avis positif": (
            "This product is amazing. Delivery was fast and customer support was very helpful."
        ),
        "Avis négatif": (
            "This is terrible. The product arrived broken and customer service never answered."
        ),
        "Avis mitigé": (
            "The product is okay, but delivery was late and the support experience could be better."
        ),
        "Texte personnalisé": ""
    }

    default_text = examples[example_choice]

    user_text = st.text_area(
        "✍️ Avis client",
        value=default_text,
        height=160,
        placeholder="Entrez ici un avis client à analyser..."
    )

    col_btn, col_info = st.columns([1, 2])

    with col_btn:
        predict_clicked = st.button("✨ Prédire le sentiment", type="primary")

    with col_info:
        st.caption(
            "La prédiction est réalisée via `POST /predict` avec le header sécurisé `X-API-Key`."
        )

    if predict_clicked:
        if not user_text.strip():
            st.warning("Veuillez saisir un avis client avant de lancer la prédiction.")
        else:
            try:
                with st.spinner("Appel de l’API de prédiction..."):
                    result = call_api_predict(api_url, api_key, user_text)

                label = result.get("label")
                prediction = result.get("prediction")
                probability_positive = result.get("probability_positive")
                probability_negative = result.get("probability_negative")

                st.divider()

                left, right = st.columns([1, 2])

                with left:
                    if label == "positive":
                        st.success("✅ Avis prédit POSITIF")
                    else:
                        st.error("❌ Avis prédit NÉGATIF")

                    st.metric("Classe prédite", str(prediction))
                    st.metric("Label", str(label))

                with right:
                    st.subheader("📊 Probabilités du modèle")

                    metric_col1, metric_col2 = st.columns(2)

                    with metric_col1:
                        if probability_positive is not None:
                            st.metric(
                                "Probabilité positive",
                                f"{probability_positive * 100:.1f} %"
                            )
                        else:
                            st.metric("Probabilité positive", "N/A")

                    with metric_col2:
                        if probability_negative is not None:
                            st.metric(
                                "Probabilité négative",
                                f"{probability_negative * 100:.1f} %"
                            )
                        else:
                            st.metric("Probabilité négative", "N/A")

                    probability_bar(
                        "Probabilité positive",
                        probability_positive,
                        "#2ecc71"
                    )

                    probability_bar(
                        "Probabilité négative",
                        probability_negative,
                        "#e74c3c"
                    )

                    if label == "positive":
                        st.info(
                            "Interprétation métier : le modèle estime que cet avis exprime "
                            "plutôt une satisfaction client."
                        )
                    else:
                        st.warning(
                            "Interprétation métier : le modèle estime que cet avis exprime "
                            "plutôt une insatisfaction ou un signal à traiter."
                        )

                with st.expander("Voir la réponse JSON de l’API"):
                    st.json(result)

            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code == 401:
                    st.error(
                        "Erreur 401 : clé API invalide ou absente. "
                        "Vérifiez la valeur du header X-API-Key dans la barre latérale."
                    )
                else:
                    st.error(f"Erreur HTTP lors de l’appel API : {e}")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Impossible de joindre l’API FastAPI. "
                    "Vérifiez que Docker Compose est lancé et que l’API est disponible sur le port 8001."
                )

            except Exception as e:
                st.error(f"Erreur inattendue : {e}")


# ==================================================
# ONGLET MONITORING
# ==================================================

with tab_monitoring:
    st.header("📈 Monitoring live de l’API")

    st.markdown(
        """
        Cette section donne une vue simple de l’état de l’API et des métriques exposées
        à Prometheus.  
        Le dashboard complet est disponible dans Grafana.
        """
    )

    health = call_api_health(api_url)
    api_is_ok = health is not None and health.get("status") == "ok"

    api_up = query_prometheus(prometheus_url, 'up{job="trustpilot-api"}')
    total_predictions = query_prometheus(
        prometheus_url,
        "sum(trustpilot_predictions_total)"
    )
    current_positive_ratio = query_prometheus(
        prometheus_url,
        "trustpilot_current_positive_ratio"
    )
    reference_positive_ratio = query_prometheus(
        prometheus_url,
        "trustpilot_reference_positive_ratio"
    )
    drift_proxy = query_prometheus(
        prometheus_url,
        "trustpilot_prediction_drift_proxy"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        if api_is_ok:
            st.success("✅ API FastAPI joignable")
        else:
            st.error("❌ API FastAPI indisponible")

    with col2:
        if api_up == 1:
            st.success("✅ Prometheus target UP")
        elif api_up == 0:
            st.error("❌ Prometheus target DOWN")
        else:
            st.warning("⚠️ Statut Prometheus indisponible")

    with col3:
        if health:
            st.success("✅ /health OK")
        else:
            st.error("❌ /health indisponible")

    st.divider()

    m1, m2, m3, m4 = st.columns(4)

    with m1:
        st.metric(
            "Total prédictions",
            "N/A" if total_predictions is None else int(total_predictions)
        )

    with m2:
        st.metric(
            "Ratio positif courant",
            safe_percent(current_positive_ratio)
        )

    with m3:
        st.metric(
            "Ratio positif référence",
            safe_percent(reference_positive_ratio)
        )

    with m4:
        st.metric(
            "Drift proxy",
            safe_percent(drift_proxy)
        )

    st.info(
        "Le drift proxy compare le ratio courant de prédictions positives "
        "au ratio positif du dataset d’entraînement. C’est un premier signal simple "
        "de changement de distribution."
    )

    st.divider()

    st.subheader("🚦 Générer du trafic de démonstration")

    st.markdown(
        """
        Ce bouton envoie automatiquement plusieurs avis de test vers l’API `/predict`.
        Cela permet d’alimenter les métriques Prometheus et de voir le dashboard Grafana évoluer.
        """
    )

    traffic_col1, traffic_col2 = st.columns([1, 2])

    with traffic_col1:
        prediction_count = st.number_input(
            "Nombre de prédictions à générer",
            min_value=1,
            max_value=500,
            value=100,
            step=10
        )

    with traffic_col2:
        traffic_mode = st.selectbox(
            "Type de trafic simulé",
            [
                "Équilibré",
                "Majoritairement positif",
                "Majoritairement négatif"
            ]
        )

        st.caption(
            "Utile pour la démo : les compteurs Prometheus repartent de zéro "
            "si le conteneur API redémarre."
        )

    if st.button("🚀 Générer des prédictions de démonstration"):
        status_placeholder = st.empty()

        with st.spinner(f"Génération de {prediction_count} prédictions..."):
            status_placeholder.info(
                "Envoi des requêtes vers l’API. Cela peut prendre quelques secondes."
            )

            demo_results = generate_demo_predictions(
                api_url=api_url,
                api_key=api_key,
                count=int(prediction_count),
                mode=traffic_mode
            )

        status_placeholder.empty()

        st.success("Génération terminée.")

        r1, r2, r3, r4 = st.columns(4)

        with r1:
            st.metric("Total envoyé", demo_results["total"])

        with r2:
            st.metric("Positifs", demo_results["positive"])

        with r3:
            st.metric("Négatifs", demo_results["negative"])

        with r4:
            st.metric("Erreurs", demo_results["errors"])

        st.info(
            "Rafraîchissez Grafana ou l’onglet Monitoring live pour voir les métriques mises à jour."
        )

    st.divider()

    st.subheader("🔗 Outils de monitoring")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.link_button("Ouvrir Grafana", grafana_url)

    with c2:
        st.link_button("Ouvrir Prometheus", prometheus_url)

    with c3:
        st.link_button("Ouvrir Swagger API", f"{api_url.rstrip('/')}/docs")

    st.caption(
        "Dans Grafana, le dashboard principal est : Trustpilot API Monitoring."
    )


# ==================================================
# ONGLET PIPELINE MLOPS
# ==================================================

with tab_pipeline:
    st.header("🧱 Vue simplifiée du pipeline MLOps")

    st.markdown(
        """
        Cette section explique la chaîne complète de manière lisible pour un profil métier
        ou data analyst.
        """
    )

    st.subheader("🔮 Chaîne de prédiction")

    st.code(
        """
Utilisateur
   ↓
Interface Streamlit
   ↓
API FastAPI sécurisée avec X-API-Key
   ↓
Modèle de sentiment Trustpilot
   ↓
Résultat : positif / négatif + probabilités
   ↓
Métriques exposées sur /metrics
   ↓
Prometheus collecte
   ↓
Grafana visualise
        """,
        language="text"
    )

    st.subheader("🧪 Chaîne d’entraînement")

    st.code(
        """
Base SQLite Trustpilot
   ↓
Airflow déclenche training_mlflow.py
   ↓
Entraînement du modèle TF-IDF + Logistic Regression
   ↓
MLflow Tracking : métriques, paramètres, dataset hash
   ↓
MLflow Registry : versionnement du modèle
   ↓
Comparaison avec le modèle best
        """,
        language="text"
    )

    st.info(
        "Airflow peut demander une authentification. "
        "Pour la démo locale, vous pouvez utiliser le compte administrateur configuré dans le conteneur."
    )

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.link_button("MLflow", mlflow_url)

    with c2:
        st.link_button("Airflow", airflow_url)

    with c3:
        st.link_button("Grafana", grafana_url)

    with c4:
        st.link_button("Prometheus", prometheus_url)


# ==================================================
# ONGLET DOCUMENTATION
# ==================================================

with tab_docs:
    st.header("📚 Documentation, limites et perspectives")

    st.subheader("✅ Ce que permet cette interface")

    st.markdown(
        """
        - Tester une prédiction via une API sécurisée.
        - Montrer le comportement du modèle à partir d’un avis client.
        - Vérifier l’état de santé de l’API.
        - Visualiser les métriques principales exposées à Prometheus.
        - Générer du trafic de démonstration pour alimenter Grafana.
        - Accéder rapidement à Grafana, MLflow, Airflow et Swagger.
        """
    )

    st.subheader("⚠️ Limites assumées")

    st.markdown(
        """
        - Le modèle est volontairement simple : TF-IDF + Logistic Regression.
        - La sécurité repose sur une clé API simple, pas sur OAuth2.
        - Le drift proxy n’est pas une détection complète du drift sur les features textuelles.
        - Le dashboard est local et non déployé sur une infrastructure cloud.
        - Les compteurs Prometheus exposés par l’API repartent de zéro si le conteneur API redémarre.
        """
    )

    st.subheader("🚀 Perspectives")

    st.markdown(
        """
        - Brancher l’API directement sur le modèle `best` du MLflow Registry.
        - Ajouter Evidently pour une détection de drift plus complète.
        - Enregistrer chaque prédiction en base de données.
        - Ajouter des alertes Grafana.
        - Déployer la stack sur une infrastructure cloud ou Kubernetes.
        """
    )

    st.success(
        "Cette interface sert de vitrine utilisateur pour présenter le projet MLOps "
        "au jury de manière claire et non technique."
    )