from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class ChatMessage(BaseModel):
    role: str = Field(..., example="user")
    parts: List[Dict[str, str]] = Field(..., example=[{"text": "Who is Anupam?"}])

class ChatRequest(BaseModel):
    question: str
    context_chunks: Optional[List[str]] = None  # Optional; generated internally
    chat_history: List[ChatMessage] = []

class ChatResponse(BaseModel):
    answer: str
