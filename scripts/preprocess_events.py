# puls-events-chatbot-intelligent-rag/scripts/preprocess_events.py
# ➡ Nettoyage et structuration avec Pandas

import json
import pandas as pd

# Chargement des données brutes
with open("data/raw_events.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)["results"]

cleaned_rows = []

for event in raw_data:
    try:
        # Protection contre les données manquantes
        title = event.get("title_fr", "")
        description = event.get("description_fr", "")
        city = event.get("location_city", "")
        date = event.get("firstdate", "")
        url = event.get("canonicalurl", "")

        # On ignore les événements sans titre ou description
        if not title or not description:
            continue

        cleaned_rows.append({
            "title": title,
            "description": description,
            "city": city,
            "date": date,
            "url": url
        })

    except Exception as e:
        # En cas de données malformées
        print("Erreur sur un événement :", e)

# Création du DataFrame Pandas
df = pd.DataFrame(cleaned_rows)

# Sauvegarde du jeu de données propre
df.to_csv("data/cleaned_events.csv", index=False)

print("Données nettoyées sauvegardées dans data/cleaned_events.csv")
print(f"{len(df)} événements valides conservés")
