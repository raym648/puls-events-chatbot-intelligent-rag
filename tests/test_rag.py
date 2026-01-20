# puls-events-chatbot-intelligent-rag/tests/test_rag.py
# ➡ Vérifie que le système répond bien à une vraie question

import os

# 🔒 Forcer le mode CI / test AVANT l'import du module
os.environ["CI"] = "true"

from app.rag_chain import generate_answer  # noqa: E402


def test_rag_response_not_empty():
    question = "Y a-t-il des concerts à Paris ?"

    answer = generate_answer(question)

    assert answer is not None
    assert isinstance(answer, str)
    assert len(answer) > 20
