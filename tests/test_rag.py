# puls-events-chatbot-intelligent-rag/tests/test_rag.py
# ➡ Vérifie que le système répond bien à une vraie question

from app.rag_chain import generate_answer


def test_rag_response_not_empty():
    question = "Y a-t-il des concerts à Paris ?"
    answer = generate_answer(question)

    assert answer is not None
    assert len(answer) > 20
