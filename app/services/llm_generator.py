import httpx
from app.core.config import GEMINI_API_KEY
from app.services.chunk_retriever import retrieve_relevant_chunks

async def generate_answer(question: str) -> str:
    context_chunks = retrieve_relevant_chunks(question)
    context_text = "\n".join(context_chunks)

    prompt = f"""
    Context:
    {context_text}

    Question: {question}

    Answer in the same language as the question:
    """

    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    body = {
        "contents": [{"parts": [{"text": prompt}]}]
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
            headers=headers,
            json=body,
        )
        response_data = response.json()

    if "candidates" in response_data:
        try:
            return response_data["candidates"][0]["content"]["parts"][0]["text"]
        except (IndexError, KeyError) as e:
            print("Error accessing parts of the response:", e)
            return "Sorry, I couldn't generate an answer."
    else:
        print("Gemini API error response:", response_data)
        return "Gemini API error. Please try again later."
