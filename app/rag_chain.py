# puls-events-chatbot-intelligent-rag/app/rag_chain.py
# ➡ Construction du moteur RAG LangChain + FAISS + Mistral

import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv

from mistralai.client import MistralClient
from langchain.schema import Document  # noqa: F401

# ===============================
# Chargement des secrets
# ===============================
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=MISTRAL_API_KEY)

# ===============================
# Chargement FAISS + métadonnées
# ===============================
index = faiss.read_index("vectorstore/faiss.index")

with open("vectorstore/faiss_store.pkl", "rb") as f:
    metadata = pickle.load(f)


# ===============================
# Fonction d’embedding (Mistral)
# ===============================
def embed_text(text: str):
    """
    Convertit un texte en vecteur numérique via le modèle Mistral
    """
    emb = client.embeddings(
        model="mistral-embed",
        input=[text]
    ).data[0].embedding

    vec = np.array([emb]).astype("float32")
    faiss.normalize_L2(vec)  # normalisation cosinus
    return vec


# ===============================
# Recherche sémantique FAISS
# ===============================
def retrieve_events(query: str, k: int = 5):
    """
    Recherche les k événements les plus pertinents pour la question utilisateur
    """
    query_vec = embed_text(query)
    scores, indices = index.search(query_vec, k)

    results = []
    for i, idx in enumerate(indices[0]):
        event = metadata[idx]
        event["score"] = float(scores[0][i])
        results.append(event)

    return results


# ===============================
# Génération RAG avec Mistral
# ===============================
def generate_answer(question: str):
    """
    Pipeline RAG :
    1. Recherche FAISS
    2. Injection du contexte
    3. Appel Mistral
    """
    retrieved_events = retrieve_events(question)

    # Construction du contexte à partir des événements
    context = ""
    for e in retrieved_events:
        context += (
            f"- {e['title']} à {e['city']} le {e['date']}\n"
            f"  Description: {e['description']}\n"
            f"  Lien: {e['url']}\n\n"
        )

    # Prompt RAG
    prompt = f"""
Tu es un assistant spécialisé dans les événements culturels.
Tu dois répondre uniquement à partir des informations fournies.

Événements disponibles :
{context}

Question utilisateur :
{question}

Réponds de façon claire, utile et naturelle.
"""

    response = client.chat(
        model="mistral-small",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
