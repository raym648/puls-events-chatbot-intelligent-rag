
# puls-events-chatbot-intelligent-rag/app/rag_service.py
# 👉 Couche Service du système RAG (Étape 5)

"""
Service central du système RAG.
Encapsule FAISS + LangChain + Mistral.
Utilisé par l’API FastAPI et par l’évaluation Ragas.
"""

from app.rag_chain import build_rag_chain


class RAGService:
    """
    Façade métier du chatbot RAG.
    """

    def __init__(self):
        self.chain = None
        self.retriever = None

    # --------------------------------------------------
    # Chargement paresseux
    # --------------------------------------------------
    def load(self):
        """
        Initialise la chaîne RAG si nécessaire.
        """
        if self.chain is None:
            self.chain = build_rag_chain()
            self.retriever = self.chain.retriever

    # --------------------------------------------------
    # Requête utilisateur
    # --------------------------------------------------
    def ask(self, question: str) -> dict:
        """
        Retourne la réponse et les contextes récupérés,
        nécessaires pour l’évaluation Ragas.
        """
        self.load()

        # 1. Récupération des documents
        docs = self.retriever.get_relevant_documents(question)
        contexts = [doc.page_content for doc in docs]

        # 2. Génération de la réponse via la chaîne RAG
        result = self.chain.invoke({"query": question})

        return {
            "answer": result["result"],
            "contexts": contexts,
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
        self.retriever = None
