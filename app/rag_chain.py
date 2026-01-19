# puls-events-chatbot-intelligent-rag/app/rag_chain.py
# ➜ RAG LangChain + FAISS + Mistral
# Conforme à l'Étape 3 (FAISS) + Étape 4 (LangChain)

import os
import pickle
from dotenv import load_dotenv

import faiss
from mistralai.client import MistralClient

from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.documents import Document
from langchain.docstore import InMemoryDocstore


# ============================================================
# 1. Variables d'environnement
# ============================================================

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


# ============================================================
# 2. Embeddings Mistral → LangChain
# ============================================================

class MistralEmbeddings(Embeddings):
    """Adaptateur LangChain pour Mistral Embeddings"""

    def embed_documents(self, texts):
        response = mistral_client.embeddings(
            model="mistral-embed",
            input=texts
        )
        return [e.embedding for e in response.data]

    def embed_query(self, text):
        return self.embed_documents([text])[0]


# ============================================================
# 3. LLM Mistral → LangChain
# ============================================================

class MistralChatLLM(BaseChatModel):
    """Adaptateur LangChain pour Mistral Chat"""

    @property
    def _llm_type(self):
        return "mistral"

    def _generate(self, messages, stop=None):
        prompt = "\n".join([m.content for m in messages])

        response = mistral_client.chat(
            model="mistral-small",
            messages=[{"role": "user", "content": prompt}]
        )

        content = response.choices[0].message.content
        return AIMessage(content=content)


# ============================================================
# 4. Chargement FAISS natif + métadonnées
# ============================================================

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "faiss_store.pkl")

# FAISS natif (celui créé à l'Étape 3)
index = faiss.read_index(INDEX_PATH)

# Métadonnées
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# Reconstruction des Documents LangChain
documents = [
    Document(
        page_content=event["description"],
        metadata={
            "title": event["title"],
            "city": event["city"],
            "date": event["date"],
            "url": event["url"],
        }
    )
    for event in metadata
]


# ============================================================
# 5. Injection du FAISS natif dans LangChain
# ============================================================

embeddings = MistralEmbeddings()

vectorstore = LC_FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(
        {i: doc for i, doc in enumerate(documents)}
    ),
    index_to_docstore_id={i: i for i in range(len(documents))}
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})


# ============================================================
# 6. Prompt RAG
# ============================================================

PROMPT = PromptTemplate(
    template="""
Tu es un assistant spécialisé dans les événements culturels.

Tu dois répondre uniquement à partir des événements suivants :

{context}

Question :
{question}

Réponds de manière claire, utile et naturelle.
""",
    input_variables=["context", "question"],
)


# ============================================================
# 7. Chaîne RAG
# ============================================================

llm = MistralChatLLM()

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=False,
)


# ============================================================
# 8. Point d’entrée utilisé par run_chatbot.py
# ============================================================

def generate_answer(question: str) -> str:
    result = qa_chain.invoke({"query": question})
    return result["result"]


# ============================================================
# 9. Factory pour l’API (Étape 5)
# ============================================================

def build_rag_chain():
    return qa_chain
