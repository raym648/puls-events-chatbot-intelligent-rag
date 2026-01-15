# puls-events-chatbot-intelligent-rag/app/main.py
# 👉 API FastAPI du système RAG (Étape 5)

from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field

from app.rag_service import RAGService
from app.security import verify_admin_token

app = FastAPI(
    title="Cultural Events RAG API",
    description=(
        "API REST pour interroger un système RAG "
        "LangChain + FAISS + Mistral"
    ),
    version="1.0.0",
)

rag_service = RAGService()


# ============================================================
# Modèles
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3)


class AnswerResponse(BaseModel):
    answer: str


# ============================================================
# Endpoints
# ============================================================

@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    """
    Interroge le moteur RAG.
    """
    try:
        answer = rag_service.ask(payload.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload", dependencies=[Depends(verify_admin_token)])
def reload_rag():
    """
    Recharge FAISS + LangChain après reconstruction offline.
    À appeler APRÈS avoir exécuté :
        python scripts/build_faiss_index.py
    """
    try:
        rag_service.reload()
        return {"status": "RAG reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
