# puls-events-chatbot-intelligent-rag/tests/test_rag.py
# ➡ Test unitaire CI-safe de build_rag_chain()

import sys
import types
from unittest.mock import MagicMock
import pickle

# ============================================================
# 1. MOCKS AVANT IMPORT (OBLIGATOIRE)
# ============================================================

# --- Mock langchain.chains.RetrievalQA -----------------------
fake_langchain = types.ModuleType("langchain")
fake_chains = types.ModuleType("langchain.chains")


class FakeRetrievalQA:
    @classmethod
    def from_chain_type(cls, *args, **kwargs):
        return cls()

    def invoke(self, *args, **kwargs):
        return {"result": "Concerts disponibles à Paris ce mois-ci."}


fake_chains.RetrievalQA = FakeRetrievalQA
fake_langchain.chains = fake_chains

sys.modules["langchain"] = fake_langchain
sys.modules["langchain.chains"] = fake_chains


# --- Mock faiss ----------------------------------------------
fake_faiss = types.ModuleType("faiss")
fake_faiss.read_index = MagicMock(return_value="FAKE_INDEX")
sys.modules["faiss"] = fake_faiss


# --- Mock pickle.load ----------------------------------------
pickle.load = MagicMock(
    return_value=[
        {
            "title": "Concert Jazz",
            "city": "Paris",
            "date": "2025-03-10",
            "description": "Un concert de jazz exceptionnel."
        }
    ]
)


# --- Mock MistralClient --------------------------------------
fake_mistral = types.ModuleType("mistralai.client")


class FakeMistralClient:
    def __init__(self, *args, **kwargs):
        pass

    def chat(self, *args, **kwargs):
        class Choice:
            message = type("msg", (), {"content": "Réponse Mistral simulée"})
        return type("resp", (), {"choices": [Choice()]})

    def embeddings(self, *args, **kwargs):
        return type(
            "resp",
            (),
            {"data": [type("d", (), {"embedding": [0.1, 0.2, 0.3]})]}
        )


fake_mistral.MistralClient = FakeMistralClient
sys.modules["mistralai.client"] = fake_mistral


# ============================================================
# 2. IMPORT DU MODULE À TESTER (APRÈS MOCKS)
# ============================================================

from app.rag_chain import build_rag_chain  # noqa: E402


# ============================================================
# 3. TEST
# ============================================================

def test_build_rag_chain_returns_answer():
    qa_chain, documents = build_rag_chain()

    # Vérifications structurelles
    assert qa_chain is not None
    assert documents is not None
    assert len(documents) == 1

    # Vérification fonctionnelle
    result = qa_chain.invoke({"query": "Y a-t-il des concerts à Paris ?"})

    assert isinstance(result, dict)
    assert "result" in result
    assert len(result["result"]) > 20
