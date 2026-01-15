# puls-events-chatbot-intelligent-rag/scripts/fetch_openagenda.py
# ➡ Récupération des événements via OpenAgenda API

import requests
import json
from datetime import datetime, timedelta, timezone

# ===============================
# Paramètres de filtrage métier
# ===============================

CITY = "Paris"                     # Ville ciblée
DAYS_PAST = 365                    # 1 an d'historique
DAYS_FUTURE = 365                  # 1 an à venir
LIMIT = 100                       # Nombre max d'événements récupérés

# ===============================
# Calcul des dates (UTC timezone-aware)
# ===============================

today = datetime.now(timezone.utc)

date_min = (today - timedelta(days=DAYS_PAST)).strftime("%Y-%m-%d")
date_max = (today + timedelta(days=DAYS_FUTURE)).strftime("%Y-%m-%d")

# ===============================
# Endpoint OpenAgenda (Opendatasoft)
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
}


print("Requête OpenAgenda en cours...")
response = requests.get(BASE_URL, params=params)

# ===============================
# Vérification de la réponse
# ===============================

if response.status_code != 200:
    raise Exception(
        f"Erreur OpenAgenda: {response.status_code} – "
        f"{response.text}"
    )

data = response.json()

# ===============================
# Sauvegarde brute des événements
# ===============================

with open("data/raw_events.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(
    f"{len(data['results'])} événements récupérés et "
    "sauvegardés dans data/raw_events.json"
)
