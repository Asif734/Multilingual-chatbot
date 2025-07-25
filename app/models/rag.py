from pydantic import BaseModel, Field
from typing import List, Dict

class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    parts: List[Dict[str, str]] = Field(..., example=[{"text": "Who is Anupam?"}])

class ChatRequest(BaseModel):
    question: str
    context_chunks: List[str]
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
