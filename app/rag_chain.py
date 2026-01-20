# puls-events-chatbot-intelligent-rag/app/rag_chain.py
# ➜ Construction de la chaîne RAG (FAISS + LangChain + Mistral)

import os
import pickle
from typing import List, Tuple

import faiss
from dotenv import load_dotenv
from mistralai.client import MistralClient

from langchain_community.vectorstores import FAISS as LC_FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.embeddings import Embeddings


# ============================================================
# 1. Configuration
# ============================================================

load_dotenv()
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

VECTORSTORE_DIR = "vectorstore"
INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss.index")
META_PATH = os.path.join(VECTORSTORE_DIR, "faiss_store.pkl")

mistral_client = MistralClient(api_key=MISTRAL_API_KEY)


# ============================================================
# 2. Embeddings (QUERY ONLY – cohérent avec FAISS offline)
# ============================================================

class MistralEmbeddings(Embeddings):
    """Embeddings uniquement pour la requête utilisateur"""

    def embed_query(self, text: str):
        response = mistral_client.embeddings(
            model="mistral-embed",
            input=[text]
        )
        return response.data[0].embedding

    def embed_documents(self, texts):
        raise NotImplementedError(
            "Les embeddings documents sont calculés offline"
        )


# ============================================================
# 3. LLM Mistral
# ============================================================

class MistralChatLLM(BaseChatModel):

    @property
    def _llm_type(self):
        return "mistral"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        prompt = "\n".join(m.content for m in messages)

        response = mistral_client.chat(
            model="mistral-small",
            messages=[{"role": "user", "content": prompt}]
        )

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(
                        content=response.choices[0].message.content
                    )
                )
            ]
        )


# ============================================================
# 4. Prompt RAG
# ============================================================

PROMPT = PromptTemplate(
    template="""
Tu es un assistant spécialisé dans les événements culturels.

À partir UNIQUEMENT des événements ci-dessous, réponds à la question.
Pour chaque événement affiché, tu DOIS :
- indiquer le titre
- indiquer la ville
- indiquer la date
- décrire brièvement l’événement

Si un événement ne correspond PAS à la question, ne l’affiche PAS.

Événements :
{context}

Question :
{question}

Réponse :
""",
    input_variables=["context", "question"],
)


# ============================================================
# 5. Factory principale
# ============================================================

def build_rag_chain() -> Tuple[RetrievalQA, List[Document]]:
    """
    Construit la chaîne RAG et retourne aussi les documents sources.
    """

    # Chargement FAISS
    index = faiss.read_index(INDEX_PATH)

    # Chargement métadonnées
    with open(META_PATH, "rb") as f:
        metadata = pickle.load(f)

    # Reconstruction Documents
    documents = [
        Document(
            page_content=(
                f"Titre : {e['title']}\n"
                f"Ville : {e['city']}\n"
                f"Date : {e['date']}\n"
                f"Description : {e['description']}"
            ),
            metadata=e
        )
        for e in metadata
    ]

    vectorstore = LC_FAISS(
        embedding_function=MistralEmbeddings(),
        index=index,
        docstore=InMemoryDocstore(
            {i: doc for i, doc in enumerate(documents)}
        ),
        index_to_docstore_id={i: i for i in range(len(documents))}
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa_chain = RetrievalQA.from_chain_type(
        llm=MistralChatLLM(),
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

    return qa_chain, documents


# ============================================================
# 6. API publique testable (utilisée par pytest & app)
# ============================================================

def generate_answer(question: str) -> str:
    """
    Génère une réponse textuelle à partir de la chaîne RAG.
    """
    if not question or not question.strip():
        return ""

    qa_chain, _ = build_rag_chain()

    result = qa_chain.invoke({"query": question})

    # RetrievalQA avec return_source_documents=True
    if isinstance(result, dict):
        return result.get("result", "")

    return str(result)
