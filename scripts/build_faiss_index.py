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
# Chargement des données vectorisées
# ===============================
df = pd.read_pickle(DATA_PATH)

# Vérification qu'il y a bien des embeddings
assert "embedding" in df.columns, "La colonne 'embedding' est manquante"
assert len(df) > 0, "Aucun événement à indexer"

# Conversion des embeddings en matrice numpy float32 (exigence FAISS)
embeddings = np.vstack(df["embedding"].values).astype("float32")

# Dimension des vecteurs
dim = embeddings.shape[1]
print(f"Dimension des vecteurs : {dim}")
print(f"Nombre d'événements à indexer : {len(embeddings)}")

# ===============================
# Création de l'index FAISS
# ===============================
# IndexFlatIP = recherche par similarité cosinus (après normalisation)
index = faiss.IndexFlatIP(dim)

# Normalisation L2 pour que le produit scalaire ≈ cosinus
faiss.normalize_L2(embeddings)

# Ajout des vecteurs à l’index
index.add(embeddings)

# Vérification que tous les événements sont bien indexés
assert index.ntotal == len(embeddings), (
    "Tous les événements n'ont pas été indexés"
)

# ===============================
# Sauvegarde de l’index
# ===============================
faiss.write_index(index, INDEX_PATH)

# ===============================
# Sauvegarde des métadonnées
# ===============================
# On stocke les informations utiles pour retrouver les événements
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
