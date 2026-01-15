# puls-events-chatbot-intelligent-rag/tests/test_preprocessing.py
# ➡ Test unitaire simple

import pandas as pd

df = pd.read_csv("data/cleaned_events.csv")


def test_events_not_empty():
    assert len(df) > 0, "Aucun événement n'a été récupéré"


def test_columns_exist():
    expected_cols = ["title", "description", "city", "date", "url"]
    for col in expected_cols:
        assert col in df.columns, f"Colonne manquante: {col}"
