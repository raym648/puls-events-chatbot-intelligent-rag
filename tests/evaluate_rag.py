# puls-events-chatbot-intelligent-rag/tests/evaluate_rag.py
# 👉 Évaluation automatique avec Ragas

"""
Évaluation automatique du système RAG avec Ragas.
"""

import sys
import os

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

from app.rag_service import RAGService

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    example["answer"] = rag.ask(example["question"])
    return example


dataset = dataset.map(generate_answer)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy]
)

print(results)
