# puls-events-chatbot-intelligent-rag/scripts/fetch_openagenda.py
# ➡ Récupération paginée des événements via OpenAgenda API

import requests
import json
from datetime import datetime, timedelta, timezone

CITY = "Paris"
DAYS_PAST = 365
DAYS_FUTURE = 365

LIMIT = 100            # ⚠️ max autorisé par OpenAgenda
MAX_EVENTS = 350       # objectif métier

today = datetime.now(timezone.utc)
date_min = (today - timedelta(days=DAYS_PAST)).strftime("%Y-%m-%d")
date_max = (today + timedelta(days=DAYS_FUTURE)).strftime("%Y-%m-%d")

print(f"Récupération des événements à {CITY}")
print(f"Période : {date_min} → {date_max}")

BASE_URL = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/"
    "datasets/evenements-publics-openagenda/records"
)

all_events = []
offset = 0

while len(all_events) < MAX_EVENTS:
    params = {
        "where": (
            f"location_city='{CITY}' AND "
            f"firstdate_begin <= '{date_max}' AND "
            f"lastdate_end >= '{date_min}'"
        ),
        "limit": LIMIT,
        "offset": offset,
        "order_by": "firstdate_begin ASC"
    }

    print(f"Requête OpenAgenda (offset={offset})...")
    response = requests.get(BASE_URL, params=params, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Erreur OpenAgenda ({response.status_code}) : {response.text}"
        )

    data = response.json()
    results = data.get("results", [])

    if not results:
        break

    all_events.extend(results)
    offset += LIMIT

print(f"{len(all_events)} événements récupérés")

with open("data/raw_events.json", "w", encoding="utf-8") as f:
    json.dump({"results": all_events}, f, ensure_ascii=False, indent=2)

print("✅ Données sauvegardées dans data/raw_events.json")
