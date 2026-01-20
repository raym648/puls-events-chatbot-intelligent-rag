# puls-events-chatbot-intelligent-rag/tests/test_rag.py
# ➡ Test unitaire du module RAG sans dépendances LangChain réelles

import sys
import types


# ============================================================
# 1. Mock minimal de langchain.chains.RetrievalQA
#    (pour éviter l'erreur d'import en CI)
# ============================================================

fake_langchain = types.ModuleType("langchain")
fake_chains = types.ModuleType("langchain.chains")


class FakeRetrievalQA:
    """Stub minimal pour satisfaire l'import"""

    @classmethod
    def from_chain_type(cls, *args, **kwargs):
        return cls()

    def invoke(self, *args, **kwargs):
        return {"result": "Réponse simulée à propos de concerts à Paris."}


fake_chains.RetrievalQA = FakeRetrievalQA
fake_langchain.chains = fake_chains

sys.modules["langchain"] = fake_langchain
sys.modules["langchain.chains"] = fake_chains


# ============================================================
# 2. Import du module à tester (maintenant sécurisé)
# ============================================================

from app.rag_chain import generate_answer  # noqa: E402


# ============================================================
# 3. Test
# ============================================================

def test_rag_response_not_empty():
    question = "Y a-t-il des concerts à Paris ?"

    answer = generate_answer(question)

    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 20
