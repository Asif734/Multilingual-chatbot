# import google.generativeai as genai
# import os
# from dotenv import load_dotenv
# from typing import List

# # Load environment variables from .env file
# load_dotenv()

# # Configure the Gemini API key
# # Ensure GEMINI_API_KEY is set in your environment or .env file
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# if not GEMINI_API_KEY:
#     raise ValueError("GEMINI_API_KEY not found. Please set it in your environment variables or a .env file.")

# genai.configure(api_key=GEMINI_API_KEY)

# async def generate_answer_with_context(question: str, context_chunks: List[str]) -> str:
#     """
#     Generates an answer to a question using the LLM, augmented by provided context.
#     The response will be in the same language as the question.
#     """
#     model = genai.GenerativeModel('gemini-2.0-flash') # Or 'gemini-pro' depending on your needs

#     context_text = "\n".join(context_chunks)

#     # Craft the prompt for the LLM.
#     # Emphasize answering in the same language as the question.
#     prompt = f"""
#     You are a helpful assistant. Use the following context to answer the question.
#     If the answer is not available in the context, politely state that you cannot find the answer in the provided information.
#     Answer in the same language as the question.

#     Context:
#     {context_text}

#     Question: {question}

#     Answer:
#     """

#     try:
#         # Use generate_content_async for asynchronous calls
#         response = await model.generate_content_async(prompt)
#         # Accessing response.text directly handles most cases.
#         # Check for safety ratings if productionizing.
#         return response.text
#     except Exception as e:
#         print(f"Error generating content from Gemini API: {e}")
#         # In a real app, you might log this error and return a more user-friendly message
#         return "Sorry, I am unable to generate an answer at this moment. Please try again later."

# if __name__ == "__main__":
#     import asyncio

#     # Test the LLM integration
#     async def test_llm():
#         print("Testing LLM with some sample context and questions...")
#         sample_context_en = [
#             "The capital of France is Paris. Paris is known for its Eiffel Tower.",
#             "Bangladesh gained independence in 1971 after a liberation war."
#         ]
#         sample_context_bn = [
#             "বাংলাদেশের রাজধানী ঢাকা। ঢাকা বুড়িগঙ্গা নদীর তীরে অবস্থিত।",
#             "মুক্তিযুদ্ধ ১৯৭১ সালে সংঘটিত হয়েছিল।"
#         ]

#         question_en = "What is the capital of France?"
#         answer_en = await generate_answer_with_context(question_en, sample_context_en)
#         print(f"\nEnglish Question: {question_en}\nEnglish Answer: {answer_en}")

#         question_bn = "বাংলাদেশের স্বাধীনতা কত সালে হয়েছিল?"
#         answer_bn = await generate_answer_with_context(question_bn, sample_context_bn)
#         print(f"\nBengali Question: {question_bn}\nBengali Answer: {answer_bn}")

#         question_no_context = "What is the highest mountain in the world?"
#         answer_no_context = await generate_answer_with_context(question_no_context, sample_context_en)
#         print(f"\nNo Context Question: {question_no_context}\nAnswer: {answer_no_context}")

#     asyncio.run(test_llm())




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

#     rag_prompt = f"""
#    You are an expert of reading bengali documents.You have be more intelligent, many things might be not in the document directly but you have to make correlation
#     and find a meaningful answer. You can answer by analysis from the given data sources on the basis of context. Suppose, a user asked
#    a question like "Who is anupam's uncle?". It is not directly mentioned in the documents but when you read context wise, you will get Shomvunath is the man 
#    who is uncle of Anupam's. If questions demand one word answer then do not explain.
#     Answer in the same language as the question.

#     Context:
#     {context_text}

#     Question: {question}

#     Answer:
#     """
    
    rag_prompt = f"""
    You are an advanced multilingual AI assistant designed to help users understand complex ideas from educational documents. You are given contextual text extracted from an academic paper (e.g., high school Bangla 1st Paper). Your job is to answer user questions based on that context.

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
- Provide a well-formed, thoughtful answer in the **same language** as the question.

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


