from fastapi import APIRouter
from app.models.rag import QueryRequest, QueryResponse
from app.services.llm_generator import generate_answer

rag_router = APIRouter()

@rag_router.post("/rag", response_model=QueryResponse)
async def rag_handler(query: QueryRequest):
    answer = await generate_answer(query.question)
    return QueryResponse(answer=answer)
