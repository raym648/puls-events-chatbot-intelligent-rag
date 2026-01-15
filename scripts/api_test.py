# puls-events-chatbot-intelligent-rag/scripts/api_test.py
# 👉 Test fonctionnel API

"""
Test fonctionnel simple de l'API REST RAG.
"""

import requests

API_URL = "http://localhost:8000"


def test_ask():
    payload = {
        "question": "Quels événements culturels sont recommandés ce week-end ?"
    }

    response = requests.post(f"{API_URL}/ask", json=payload)
    assert response.status_code == 200
    print("Réponse RAG :", response.json())


def test_rebuild():
    headers = {
        "X-Admin-Token": "CHANGE_ME"
    }

    response = requests.post(f"{API_URL}/rebuild", headers=headers)
    assert response.status_code == 200
    print(response.json())


if __name__ == "__main__":
    test_ask()
