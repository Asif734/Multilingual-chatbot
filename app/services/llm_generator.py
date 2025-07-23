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
    chat_history: List[Dict[str, Any]] = None # New parameter for chat history
) -> str:
    """
    Generates an answer to a question using the LLM, augmented by provided context and chat history.
    The response will be in the same language as the question.
    """
    model = genai.GenerativeModel('gemini-1.5-flash') # Or 'gemini-pro'

    context_text = "\n".join(context_chunks)

    # Prepare messages for the LLM, including history and the current prompt
    messages = []

    # Add previous chat history if available
    if chat_history:
        messages.extend(chat_history)

    # Add the current RAG prompt as the latest user turn
    # It's important to put the context and question in the user's turn for RAG
    rag_prompt = f"""
    You are a helpful assistant. Use the following context to answer the question.
    If the answer is not available in the context, politely state that you cannot find the answer in the provided information.
    Answer in the same language as the question.

    Context:
    {context_text}

    Question: {question}

    Answer:
    """
    messages.append({"role": "user", "parts": [{"text": rag_prompt}]})

    try:
        # Use generate_content_async for asynchronous calls
        # The entire conversation history is passed in 'contents'
        response = await model.generate_content_async(messages)
        return response.text
    except Exception as e:
        print(f"Error generating content from Gemini API: {e}")
        return "Sorry, I am unable to generate an answer at this moment. Please try again later."

if __name__ == "__main_": # Changed to avoid automatic run during import
    import asyncio

    async def test_llm_with_history():
        print("Testing LLM with sample context and chat history...")
        sample_context = [
            "The capital of France is Paris. Paris is known for its Eiffel Tower.",
            "Bangladesh gained independence in 1971 after a liberation war."
        ]

        # Initial turn
        first_question = "What is the capital of France?"
        initial_history = []
        first_answer = await generate_answer_with_context(first_question, sample_context, initial_history)
        print(f"\nFirst Question: {first_question}\nFirst Answer: {first_answer}")

        # Second turn, building on history
        # History should include the previous user query and model response
        updated_history = [
            {"role": "user", "parts": [{"text": f"Context: {sample_context[0]}\nQuestion: {first_question}"}]},
            {"role": "model", "parts": [{"text": first_answer}]}
        ]
        second_question = "What is it famous for?" # Referring to Paris from previous turn
        second_answer = await generate_answer_with_context(second_question, sample_context, updated_history)
        print(f"\nSecond Question: {second_question}\nSecond Answer: {second_answer}")

        # Bengali turn
        question_bn = "বাংলাদেশের স্বাধীনতা কত সালে হয়েছিল?"
        answer_bn = await generate_answer_with_context(question_bn, sample_context, [])
        print(f"\nBengali Question: {question_bn}\nBengali Answer: {answer_bn}")

    asyncio.run(test_llm_with_history())