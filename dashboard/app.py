# puls-events-chatbot-intelligent-rag/dashboard/app.py
# 📊 Dashboard de monitoring du système RAG

import os
import requests
import streamlit as st

# ============================================================
# Configuration
# ============================================================

API_URL = os.getenv("RAG_API_URL")

if not API_URL:
    st.error("❌ RAG_API_URL not configured")
    st.stop()

st.set_page_config(page_title="RAG Monitor", layout="wide")
st.title("📊 RAG Monitoring Dashboard")

# ============================================================
# Requête utilisateur
# ============================================================

query = st.text_input("Posez une question sur les événements culturels :")

if st.button("🔍 Interroger le RAG") and query:
    with st.spinner("Interrogation du moteur RAG..."):
        try:
            response = requests.post(
                f"{API_URL}/ask",
                json={"question": query},
                timeout=60
            )

            if response.status_code != 200:
                st.error(f"❌ Erreur API ({response.status_code})")
                st.code(response.text)
                st.stop()

            data = response.json()

            # Validation minimale de la réponse
            answer = data.get("answer")
            contexts = data.get("contexts", [])

            if not answer:
                st.warning("⚠️ Aucune réponse générée")
                st.stop()

            # ====================================================
            # Affichage de la réponse
            # ====================================================
            st.success("✅ Réponse générée")
            st.markdown("### 🧠 Réponse du chatbot")
            st.write(answer)

            # ====================================================
            # Affichage des documents sources
            # ====================================================
            st.markdown("### 📚 Événements utilisés comme contexte")

            if not contexts:
                st.info("Aucun document de contexte n’a été utilisé.")
            else:
                for i, ctx in enumerate(contexts, start=1):
                    with st.expander(f"Événement {i}"):
                        st.text(ctx)

        except requests.exceptions.RequestException as e:
            st.error("❌ Impossible de contacter l’API RAG")
            st.exception(e)

        except ValueError:
            st.error("❌ Réponse JSON invalide retournée par l’API")
            st.code(response.text)

# ============================================================
# Section Reload FAISS
# ============================================================

st.divider()
st.markdown("### 🔄 Administration du vector store")

if st.button("Recharger le vector store FAISS"):
    admin_token = os.getenv("ADMIN_TOKEN")

    if not admin_token:
        st.error("❌ ADMIN_TOKEN manquant")
        st.stop()

    with st.spinner("Rechargement du vector store..."):
        try:
            response = requests.post(
                f"{API_URL}/reload",
                headers={"Authorization": f"Bearer {admin_token}"},
                timeout=300
            )

            if response.status_code != 200:
                st.error(f"❌ Échec du reload ({response.status_code})")
                st.code(response.text)
                st.stop()

            st.success("✅ Vector store rechargé avec succès")

        except requests.exceptions.RequestException as e:
            st.error("❌ Échec de l’appel au endpoint /reload")
            st.exception(e)

# ============================================================
# Information RAGAS
# ============================================================

st.divider()
st.info(
    "ℹ️ Les métriques RAGAS ne sont pas exposées via l’API, "
    "mais les contextes retournés permettent une évaluation offline."
)
