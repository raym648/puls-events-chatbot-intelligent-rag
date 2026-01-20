# puls-events-chatbot-intelligent-rag/scripts/preprocess_events.py
# ➡ Nettoyage et structuration des événements OpenAgenda (version corrigée)

import json
import pandas as pd
from datetime import datetime  # noqa: F401

# ===============================
# Chargement des données brutes
# ===============================

with open("data/raw_events.json", "r", encoding="utf-8") as f:
    data = json.load(f)

if "results" not in data or not isinstance(data["results"], list):
    raise ValueError("Structure invalide du fichier raw_events.json")

raw_data = data["results"]

cleaned_rows = []

# ===============================
# Nettoyage événement par événement
# ===============================

for event in raw_data:
    try:
        title = (event.get("title_fr") or "").strip()
        description = (event.get("description_fr") or "").strip()
        city = (event.get("location_city") or "").strip()
        url = event.get("canonicalurl") or ""

        # Champs OpenAgenda fiables
        daterange_fr = event.get("daterange_fr")
        date_start = event.get("firstdate_begin")
        date_end = event.get("firstdate_end")

        # Filtres qualité stricts
        if not title or not description:
            continue
        if not city:
            continue
        if not daterange_fr or not date_start:
            continue

        cleaned_rows.append({
            "title": title,
            "description": description,
            "city": city,

            # 🧠 Date lisible (LLM / RAG)
            "date": daterange_fr,

            # ⚙️ Dates techniques (ISO)
            "date_start": date_start,
            "date_end": date_end,

            "url": url
        })

    except Exception as e:
        print("Erreur sur un événement :", e)

# ===============================
# Création du DataFrame
# ===============================

df = pd.DataFrame(cleaned_rows)

if df.empty:
    raise ValueError("Aucun événement valide après nettoyage")

# ===============================
# Tri temporel (événements à venir en priorité)
# ===============================

df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
df = df.sort_values(by="date_start", ascending=True)

# ===============================
# Sauvegarde
# ===============================

df.to_csv("data/cleaned_events.csv", index=False)

print("Données nettoyées sauvegardées dans data/cleaned_events.csv")
print(f"{len(df)} événements valides conservés")
