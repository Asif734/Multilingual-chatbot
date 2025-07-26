


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn
import os
import asyncio
from typing import List, Dict, Any, Optional

# Import your modules
from app.services.retriever import initialize_retriever_from_text, retrieve_relevant_chunks, rag_vector_store
from app.services.llm_generator import generate_answer_with_context
from app.models.rag import ChatMessage, ChatResponse, ChatRequest

# --- Configuration ---
TEXT_PATH = "/Users/asif/vscode/Multilingual_AI_Assistant_RAG/app/data/extracted_text_from_HSC26_Bangla1st-Paper.txt"
RAG_INDEX_PATH = "rag_index"

# Initialize FastAPI app
app = FastAPI(
    title="Multilingual RAG Chatbot",
    description="A chatbot that answers questions in multiple languages based on a text knowledge base using RAG and LLM, with short-term memory."
)

# --- Startup Event Handler ---
@app.on_event("startup")
async def startup_event():
    """
    Initializes the RAG retriever when the FastAPI application starts up,
    using extracted text instead of PDF.
    """
    print("FastAPI application startup: Initializing RAG from extracted text...")
    try:
        if not os.path.exists(TEXT_PATH):
            raise FileNotFoundError(f"Text file not found at {TEXT_PATH}")

        initialize_retriever_from_text(text_path=TEXT_PATH, index_path=RAG_INDEX_PATH)
        print("RAG initialized successfully from text.")
    except Exception as e:
        print(f"Failed to initialize RAG components on startup: {e}")
        raise

# --- API Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_with_pdf(request: ChatRequest):
    """
    Receives a question, retrieves relevant context from the knowledge base,
    and generates an answer using the LLM, considering chat history.
    """
    if rag_vector_store is None or not rag_vector_store.is_built:
        msg = "RAG system not initialized properly."
        print(msg)
        raise HTTPException(status_code=503, detail=msg)

    try:
        print(f"Received question: {request.question}")

        relevant_chunks = retrieve_relevant_chunks(request.question, k=3)
        print(f"Retrieved {len(relevant_chunks)} chunks")


        gemini_chat_history = []
        if request.chat_history:
            for msg in request.chat_history:
                gemini_chat_history.append({"role": msg.role, "parts": msg.parts})

        answer = await generate_answer_with_context(
            question=request.question,
            context_chunks=relevant_chunks,
            chat_history=gemini_chat_history
        )
        print(f"Generated answer: {answer}")

        return ChatResponse(answer=answer, retrieved_context=relevant_chunks)

    except Exception as e:
        print(f"Error during chat: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# --- Health Check Endpoint ---
@app.get("/health")
async def health_check():
    if rag_vector_store is None:
        return {"status": "error", "rag_initialized": False, "message": "rag_vector_store is None"}
    return {"status": "ok", "rag_initialized": rag_vector_store.is_built}

# --- Entry Point ---
if __name__ == "__main__":
    if not os.path.exists(TEXT_PATH):
        print(f"Error: The text file '{TEXT_PATH}' was not found.")
    else:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


