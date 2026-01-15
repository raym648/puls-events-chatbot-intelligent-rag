# puls-events-chatbot-intelligent-rag/scripts/embed_events.py
# ➡ Vectorisation des descriptions avec Mistral

import os
import pandas as pd
from mistralai.client import MistralClient
from dotenv import load_dotenv

# Chargement de la clé Mistral depuis .env
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

client = MistralClient(api_key=MISTRAL_API_KEY)

# Chargement des événements nettoyés
df = pd.read_csv("data/cleaned_events.csv")

texts = df["description"].tolist()

print("Génération des embeddings avec Mistral...")

# Appel API Mistral pour créer des vecteurs
embeddings = client.embeddings(
    model="mistral-embed",
    input=texts
)

# Ajout des vecteurs au DataFrame
df["embedding"] = [vec.embedding for vec in embeddings.data]

# Sauvegarde finale prête pour FAISS
df.to_pickle("data/cleaned_events_with_embeddings.pkl")

print(
    "Données vectorisées sauvegardées dans "
    "data/cleaned_events_with_embeddings.pkl"
)
