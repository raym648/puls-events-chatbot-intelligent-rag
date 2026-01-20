# puls-events-chatbot-intelligent-rag/scripts/fetch_openagenda.py
# ➡ Récupération des événements via OpenAgenda API

import requests
import json
from datetime import datetime, timedelta, timezone

# ===============================
# Paramètres de filtrage métier
# ===============================

CITY = "Paris"          # Ville ciblée
DAYS_PAST = 365         # Historique
DAYS_FUTURE = 365       # À venir
LIMIT = 100             # Limite OpenAgenda (sans pagination)

# ===============================
# Calcul des dates (UTC, ISO)
# ===============================

today = datetime.now(timezone.utc)

date_min = (today - timedelta(days=DAYS_PAST)).strftime("%Y-%m-%d")
date_max = (today + timedelta(days=DAYS_FUTURE)).strftime("%Y-%m-%d")

print(f"Récupération des événements à {CITY}")
print(f"Période : {date_min} → {date_max}")

# ===============================
# Endpoint OpenAgenda
# ===============================

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/evenements-publics-openagenda/records"
)

params = {
    "where": (
        f"location_city='{CITY}' AND "
        f"firstdate_begin <= '{date_max}' AND "
        f"lastdate_end >= '{date_min}'"
    ),
    "limit": LIMIT,
    "order_by": "firstdate_begin ASC"
}

# ===============================
# Appel API
# ===============================

print("Requête OpenAgenda en cours...")
response = requests.get(BASE_URL, params=params, timeout=30)

if response.status_code != 200:
    raise RuntimeError(
        f"Erreur OpenAgenda ({response.status_code}) : "
        f"{response.text}"
    )

data = response.json()

# ===============================
# Validation minimale des données
# ===============================

if "results" not in data or not isinstance(data["results"], list):
    raise ValueError("Structure inattendue de la réponse OpenAgenda")

if len(data["results"]) == 0:
    print("⚠️ Aucun événement retourné par OpenAgenda")

# ===============================
# Sauvegarde brute
# ===============================

with open("data/raw_events.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(
    f"{len(data['results'])} événements récupérés "
    f"et sauvegardés dans data/raw_events.json"
)
