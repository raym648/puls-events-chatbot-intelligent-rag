# puls-events-chatbot-intelligent-rag/scripts/build_faiss_index.py
# ➡ Construit l’index FAISS + sauvegarde des métadonnées

import os
import pickle
import numpy as np
import pandas as pd
import faiss

# ===============================
# Paramètres
# ===============================

DATA_PATH = "data/cleaned_events_with_embeddings.pkl"
VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "faiss_store.pkl")

os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ===============================
# Chargement des données
# ===============================

df = pd.read_pickle(DATA_PATH)

required_columns = [
    "title",
    "description",
    "city",
    "date",
    "url",
    "embedding",
]

missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le dataset : {missing}")

if len(df) == 0:
    raise ValueError("Aucun événement à indexer")

# ===============================
# Préparation des embeddings
# ===============================

embeddings = np.vstack(df["embedding"].values)

if embeddings.dtype != np.float32:
    embeddings = embeddings.astype("float32")

# Normalisation L2 (cosine similarity)
faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]

print(f"Dimension des vecteurs : {dim}")
print(f"Nombre d'événements à indexer : {len(embeddings)}")

# ===============================
# Création de l’index FAISS
# ===============================

index = faiss.IndexFlatIP(dim)
index.add(embeddings)

if index.ntotal != len(embeddings):
    raise RuntimeError("Erreur : tous les événements n'ont pas été indexés")

# ===============================
# Sauvegarde de l’index
# ===============================

faiss.write_index(index, INDEX_PATH)

# ===============================
# Sauvegarde des métadonnées
# ===============================
# ⚠️ Les dates sont conservées telles quelles
# pour éviter toute transformation destructrice

metadata_columns = [
    "title",
    "description",
    "city",
    "date",
    "url",
]

metadata = df[metadata_columns].to_dict(orient="records")

with open(META_PATH, "wb") as f:
    pickle.dump(metadata, f)

print("Index FAISS et métadonnées sauvegardés dans vectorstore/")
