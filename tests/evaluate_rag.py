# puls-events-chatbot-intelligent-rag/tests/evaluate_rag.py
# 👉 Évaluation automatique avec Ragas

"""
Évaluation automatique du système RAG avec Ragas.
"""

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from app.rag_service import RAGService


rag = RAGService()

# Jeu de test annoté manuellement
data = {
    "question": [
        "Quels événements culturels à Paris ?"
    ],
    "ground_truth": [
        "Liste d'événements culturels parisiens pertinents."
    ]
}

dataset = Dataset.from_dict(data)


def generate_answer(example):
    output = rag.ask(example["question"])
    example["answer"] = output["answer"]
    example["contexts"] = output["contexts"]
    return example


dataset = dataset.map(generate_answer)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(results)
