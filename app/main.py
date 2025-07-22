from fastapi import FastAPI
from app.api.v1.endpoints import rag_router


app= FastAPI()
app = FastAPI(title="Multilingual RAG System")
app.include_router(rag_router, prefix="/api/v1", tags=["RAG"])
