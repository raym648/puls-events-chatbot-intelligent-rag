# puls-events-chatbot-intelligent-rag/tests/test_api.py

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ask():
    response = client.post(
        "/ask",
        json={"question": "Quels événements culturels ce week-end ?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json()
