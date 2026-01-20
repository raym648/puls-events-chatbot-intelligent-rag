# puls-events-chatbot-intelligent-rag/scripts/embed_events.py
# ➡ Vectorisation des événements avec Mistral (version corrigée)

import os
import pandas as pd
from mistralai.client import MistralClient
from dotenv import load_dotenv

# ===============================
# Chargement des variables d’environnement
# ===============================

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise EnvironmentError("MISTRAL_API_KEY manquante")

client = MistralClient(api_key=MISTRAL_API_KEY)

# ===============================
# Chargement des événements nettoyés
# ===============================

df = pd.read_csv("data/cleaned_events.csv")

required_columns = ["title", "description", "city", "date", "url"]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans cleaned_events.csv : {missing}")  # noqa: E501

# ===============================
# Construction des textes à vectoriser
# ===============================
# ⚠️ IMPORTANT :
# On inclut volontairement les dates et la ville
# pour éviter toute hallucination temporelle côté RAG

texts = [
    f"Titre : {row.title}\n"
    f"Ville : {row.city}\n"
    f"Date : {row.date}\n"
    f"Description : {row.description}"
    for row in df.itertuples(index=False)
]

print("Génération des embeddings avec Mistral...")

# ===============================
# Appel API Mistral
# ===============================

embeddings_response = client.embeddings(
    model="mistral-embed",
    input=texts
)

# Vérification de cohérence
if len(embeddings_response.data) != len(df):
    raise RuntimeError("Nombre d’embeddings différent du nombre d’événements")

# ===============================
# Ajout des embeddings au DataFrame
# ===============================

df["embedding"] = [item.embedding for item in embeddings_response.data]

# ===============================
# Sauvegarde finale
# ===============================

df.to_pickle("data/cleaned_events_with_embeddings.pkl")

print(
    "Embeddings générés et sauvegardés dans "
    "data/cleaned_events_with_embeddings.pkl"
)
