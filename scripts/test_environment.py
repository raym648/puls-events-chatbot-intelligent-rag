# puls-events-chatbot-intelligent-rag/scripts/test_environment.py

# ===============================
# Test du bon fonctionnement de l'environnement RAG
# ===============================

import faiss  # noqa: F401
from langchain_community.vectorstores import FAISS  # noqa: F401
from langchain_community.embeddings import HuggingFaceEmbeddings  # noqa: F401
from mistralai.client import MistralClient  # noqa: F401

print("Python OK")
print("FAISS OK")
print("LangChain FAISS OK")
print("HuggingFace Embeddings OK")
print("Mistral Client OK")
print("Tous les composants sont correctement installés 🎉")
