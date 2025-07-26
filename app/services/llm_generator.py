




import google.generativeai as genai
import os
from dotenv import load_dotenv
from typing import List, Dict, Any


print(f"DEBUG: llm_model.py is running from: {os.path.abspath(__file__)}")
# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")

genai.configure(api_key=GEMINI_API_KEY)

async def generate_answer_with_context(
    question: str,
    context_chunks: List[str],
    chat_history: List[Dict[str, Any]] = None  # Will ignore for now
) -> str:
    model = genai.GenerativeModel('gemini-2.5-flash')

    context_text = "\n".join(context_chunks or [])

    
    rag_prompt = f"""
    You are an advanced multilingual AI assistant designed to help users understand complex ideas from educational documents.
      You are given contextual text extracted from an academic paper (e.g., high school Bangla 1st Paper). 
      Your job is to answer user questions based on that context and by analysing the whole chunk.

🧠 Important:
- The answer is not directly quoted in the text.
- You must use **semantic reasoning**, draw **logical inferences**, and identify **relationships or implications** within the content to construct an answer.
- Use your natural language understanding abilities to bridge gaps where information is implied but not stated.

🌐 Language Rule:
- If the user asks the question in English, respond in English.
- If the question is written in Bengali (বাংলা), respond in Bengali.
- Maintain the tone and clarity expected in academic environments.

📚 Instruction:
Given the following:
1. A set of **relevant context chunks** (retrieved using semantic similarity).
2. A **user question** (which may be in English or Bengali).

You must:
- Analyze the context thoroughly.
- Use reasoning and understanding of meaning, not surface-level matching.
- Provide a well-formed, thoughtful, focused answer, most preferable one or two word answer in the **same language** as the question.

---

    Context:
    {context_text}

    Question: {question}

    Answer:
    """

    # Ignore chat_history and send only current user message
    messages = [{"role": "user", "parts": [{"text": rag_prompt}]}]

    try:
        response = await model.generate_content_async(messages)
        return response.text
    except Exception as e:
        print(f"Error generating content from Gemini API: {e}")
        return "Sorry, I am unable to generate an answer at this moment. Please try again later."


