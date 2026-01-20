# puls-events-chatbot-intelligent-rag/app/rag_service.py
# 👉 Couche Service du système RAG (Étape 5)

# puls-events-chatbot-intelligent-rag/app/rag_service.py
# 👉 Couche Service du système RAG

"""
Service central du système RAG.
Façade métier entre l’API FastAPI et la chaîne RAG.
Compatible évaluation Ragas (answer + contexts).
"""

from typing import List, Dict, Any
from app.rag_chain import build_rag_chain


class RAGService:
    """
    Façade métier du chatbot RAG.
    """

    def __init__(self):
        self.qa_chain = None

    # --------------------------------------------------
    # Chargement paresseux
    # --------------------------------------------------
    def load(self):
        """
        Initialise la chaîne RAG si nécessaire.
        """
        if self.qa_chain is None:
            self.qa_chain, _ = build_rag_chain()

    # --------------------------------------------------
    # Requête utilisateur
    # --------------------------------------------------
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Exécute une requête RAG et retourne :
        - la réponse générée
        - les contextes utilisés (pour audit / Ragas)
        """
        self.load()

        result = self.qa_chain.invoke({"query": question})

        answer = result["result"]

        source_docs = result.get("source_documents", [])

        contexts: List[str] = [
            doc.page_content
            for doc in source_docs
        ]

        return {
            "answer": answer,
            "contexts": contexts,
        }

    # --------------------------------------------------
    # Rechargement du FAISS (admin)
    # --------------------------------------------------
    def reload(self):
        """
        Force la reconstruction complète du RAG.
        À appeler après rebuild FAISS offline.
        """
        self.qa_chain = None
