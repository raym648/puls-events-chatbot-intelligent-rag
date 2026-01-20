# puls-events-chatbot-intelligent-rag/scripts/embed_events.py
# ➡ Vectorisation des événements avec Mistral (batching safe)

import os
import pandas as pd
from mistralai.client import MistralClient
from dotenv import load_dotenv
from tqdm import tqdm


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
    raise ValueError(
        f"Colonnes manquantes dans cleaned_events.csv : {missing}"
    )

# ===============================
# Construction des textes à vectoriser
# ===============================


texts = [
    f"Titre : {row.title}\n"
    f"Ville : {row.city}\n"
    f"Date : {row.date}\n"
    f"Description : {row.description}"
    for row in df.itertuples(index=False)
]

print(f"Génération des embeddings avec Mistral ({len(texts)} événements)")


# ===============================
# Batching Mistral (OBLIGATOIRE)
# ===============================

BATCH_SIZE = 32
all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Embeddings"):
    batch_texts = texts[i: i + BATCH_SIZE]

    response = client.embeddings(
        model="mistral-embed",
        input=batch_texts
    )

    batch_embeddings = [item.embedding for item in response.data]
    all_embeddings.extend(batch_embeddings)

# ===============================
# Vérification de cohérence
# ===============================

if len(all_embeddings) != len(df):
    raise RuntimeError(
        f"Incohérence embeddings ({len(all_embeddings)}) "
        f"vs événements ({len(df)})"
    )

# ===============================
# Sauvegarde finale
# ===============================

df["embedding"] = all_embeddings
df.to_pickle("data/cleaned_events_with_embeddings.pkl")

print(
    "✅ Embeddings générés et sauvegardés dans "
    "data/cleaned_events_with_embeddings.pkl"
)
