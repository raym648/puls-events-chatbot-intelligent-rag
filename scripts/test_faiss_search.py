# puls-events-chatbot-intelligent-rag/scripts/test_faiss_search.py
# ➡ Test de recherche sémantique sur l’index

import os
import pickle
import numpy as np
import faiss
from mistralai.client import MistralClient
from dotenv import load_dotenv

# ===============================
# Chargement des secrets
# ===============================
load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
client = MistralClient(api_key=MISTRAL_API_KEY)

# ===============================
# Chargement de l’index FAISS
# ===============================
index = faiss.read_index("vectorstore/faiss.index")

with open("vectorstore/faiss_store.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"{index.ntotal} événements chargés dans l’index FAISS")

# ===============================
# Question test
# ===============================
query = "concert de jazz ce week-end à Paris"
print(f"Requête : {query}")

# Création de l'embedding de la requête via Mistral
query_embedding = client.embeddings(
    model="mistral-embed",
    input=[query]
).data[0].embedding

# Conversion en vecteur numpy
query_vector = np.array([query_embedding]).astype("float32")

# Normalisation pour cohérence avec l’index
faiss.normalize_L2(query_vector)

# ===============================
# Recherche des k meilleurs résultats
# ===============================
k = 5
scores, indices = index.search(query_vector, k)

print("\nRésultats :\n")

for rank, idx in enumerate(indices[0]):
    event = metadata[idx]
    score = scores[0][rank]
    print(f"{rank+1}. {event['title']} ({event['city']}, {event['date']})")
    print(f"   Score de similarité: {score:.3f}")
    print(f"   {event['url']}\n")
