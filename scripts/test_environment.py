# puls-events-chatbot-intelligent-rag/scripts/test_environment.py

# ===============================
# Test du bon fonctionnement de l'environnement RAG
# ===============================

print("Python OK")

# Test FAISS
import faiss
print("FAISS OK")

# Test LangChain FAISS wrapper
from langchain.vectorstores import FAISS
print("LangChain FAISS OK")

# Test embeddings HuggingFace
from langchain.embeddings import HuggingFaceEmbeddings
print("HuggingFace Embeddings OK")

# Test client Mistral
from mistralai.client import MistralClient
print("Mistral Client OK")

print("Tous les composants sont correctement installés 🎉")
