# from fastapi import FastAPI, HTTPException
# from pydantic import BaseModel
# import uvicorn
# import os
# import asyncio
# from typing import List


# # Import your modules
# from app.services.retriever import initialize_retriever, retrieve_relevant_chunks
# from app.services.llm_generator import generate_answer_with_context

# # --- Configuration ---
# PDF_PATH = r"app/data/HSC26-Bangla1st-Paper.pdf" # Ensure this PDF is in the same directory as main.py
# RAG_INDEX_PATH = "rag_index" # Path to save/load FAISS index and chunks

# # Initialize FastAPI app
# app = FastAPI(
#     title="Multilingual RAG Chatbot",
#     description="A chatbot that answers questions in multiple languages based on a PDF knowledge base using RAG and LLM."
# )

# # --- Pydantic Models for Request/Response ---
# class ChatRequest(BaseModel):
#     question: str

# class ChatResponse(BaseModel):
#     answer: str
#     retrieved_context: List[str]

# # --- Startup Event Handler ---
# @app.on_event("startup")
# async def startup_event():
#     """
#     Initializes the RAG retriever when the FastAPI application starts up.
#     This ensures the PDF is processed and the vector store is ready before requests come in.
#     """
#     print("FastAPI application startup: Initializing RAG components...")
#     try:
#         initialize_retriever(pdf_path=PDF_PATH, index_path=RAG_INDEX_PATH)
#         print("RAG components initialized successfully.")
#     except Exception as e:
#         print(f"Failed to initialize RAG components on startup: {e}")
#         # Depending on criticality, you might want to gracefully shut down or prevent requests
#         # For this example, we'll let it try to run but subsequent calls might fail.

# # --- API Endpoint ---
# @app.post("/chat", response_model=ChatResponse)
# async def chat_with_pdf(request: ChatRequest):
#     """
#     Receives a question, retrieves relevant context from the PDF,
#     and generates an answer using the LLM.
#     """
#     try:
#         # 1. Retrieve relevant chunks
#         relevant_chunks = retrieve_relevant_chunks(request.question, k=3)
#         if not relevant_chunks:
#             # Fallback if no relevant chunks are found, LLM can still try to answer,
#             # but it's better to provide a specific message or try to answer generally.
#             print("No relevant chunks found for the query.")
#             # return ChatResponse(answer="I could not find relevant information for your question in the document.", retrieved_context=[])

#         # 2. Generate answer using LLM with context
#         answer = await generate_answer_with_context(request.question, relevant_chunks)

#         return ChatResponse(answer=answer, retrieved_context=relevant_chunks)

#     except Exception as e:
#         print(f"An error occurred during chat processing: {e}")
#         raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# # --- Health Check Endpoint (Optional but recommended) ---
# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint to ensure the API is running.
#     """
#     return {"status": "ok", "rag_initialized": rag_vector_store.is_built} # assuming rag_vector_store is globally accessible from retriever


# if __name__ == "__main__":
#     # Ensure the PDF is present for local execution
#     if not os.path.exists(PDF_PATH):
#         print(f"Error: The PDF file '{PDF_PATH}' was not found. Please place it in the same directory as main.py.")
#         print("Exiting. Cannot start the application without the data source.")
#     else:
#         # Run the FastAPI application using Uvicorn
#         # The host='0.0.0.0' makes it accessible externally (e.g., in Docker)
#         # reload=True is good for development, automatically reloads on code changes
#         uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)



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
TEXT_PATH = r"C:\Users\Asif\VSCODE\Multilingual_AI_Assistant_RAG\extracted_text_from_HSC26_Bangla1st-Paper.txt"
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
        relevant_chunks = retrieve_relevant_chunks(request.question, k=3)

        gemini_chat_history = []
        if request.chat_history:
            for msg in request.chat_history:
                gemini_chat_history.append({"role": msg.role, "parts": msg.parts})

        answer = await generate_answer_with_context(
            question=request.question,
            context_chunks=relevant_chunks,
            chat_history=gemini_chat_history
        )

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
