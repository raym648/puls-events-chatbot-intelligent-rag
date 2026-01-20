# puls-events-chatbot-intelligent-rag/app/main.py
# 👉 API FastAPI du système RAG (Étape 5)

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import List

from app.rag_service import RAGService
from app.security import verify_admin_token

app = FastAPI(
    title="Cultural Events RAG API",
    description=(
        "API REST pour interroger un système RAG "
        "basé sur FAISS + Mistral"
    ),
    version="1.0.0",
)

# Initialisation différée du service RAG
rag_service: RAGService | None = None


# ============================================================
# Modèles API
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)


class AnswerResponse(BaseModel):
    answer: str
    contexts: List[str]


# ============================================================
# Lifecycle
# ============================================================

@app.on_event("startup")
def load_rag():
    """
    Chargement initial du moteur RAG au démarrage de l’API.
    """
    global rag_service
    rag_service = RAGService()


# ============================================================
# Endpoints
# ============================================================

@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    """
    Interroge le moteur RAG.
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized"
        )

    try:
        response = rag_service.ask(payload.question)

        # Sécurité contractuelle
        if not isinstance(response, dict):
            raise ValueError("Invalid RAG response format")

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload", dependencies=[Depends(verify_admin_token)])
def reload_rag():
    """
    Recharge FAISS + métadonnées après reconstruction offline.
    À appeler APRÈS :
        python scripts/build_faiss_index.py
    """
    if rag_service is None:
        raise HTTPException(
            status_code=503,
            detail="RAG service not initialized"
        )

    try:
        rag_service.reload()
        return {"status": "RAG reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
