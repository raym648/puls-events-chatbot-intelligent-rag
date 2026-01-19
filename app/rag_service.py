# puls-events-chatbot-intelligent-rag/app/rag_service.py
# 👉 Couche Service du système RAG (Étape 5)

"""
Service central du système RAG.
Encapsule FAISS + LangChain + Mistral.
Utilisé par l’API FastAPI.
"""

from app.rag_chain import build_rag_chain


class RAGService:
    """
    Façade métier du chatbot RAG.
    """

    def __init__(self):
        self.chain = None

    # --------------------------------------------------
    # Chargement paresseux
    # --------------------------------------------------
    def load(self):
        """
        Initialise la chaîne RAG si nécessaire.
        """
        if self.chain is None:
            self.chain = build_rag_chain()

    # --------------------------------------------------
    # Requête utilisateur
    # --------------------------------------------------
    def ask(self, question: str) -> dict:
        """
        Returns answer and retrieved contexts for evaluation.
        """
        retrieved_docs = self.retriever.get_relevant_documents(question)

        contexts = [doc.page_content for doc in retrieved_docs]

        answer = self.llm.generate_answer(question, contexts)

        return {
            "answer": answer,
            "contexts": contexts
        }

    # --------------------------------------------------
    # Rechargement du FAISS (après rebuild)
    # --------------------------------------------------
    def reload(self):
        """
        Force le rechargement du FAISS et du RAG.
        À appeler après avoir exécuté build_faiss_index.py.
        """
        self.chain = None
