import asyncio
from services.retriever import retrieve_relevant_chunks
from services.llm_generator import generate_answer_with_context
from rag_evaluation import run_sample_tests

sample_test_cases = [
    ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
    ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
    ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর")
]

run_sample_tests(
    sample_test_cases,
    retriever_func=retrieve_relevant_chunks,
    answer_generator_func=lambda q, c: asyncio.run(generate_answer_with_context(q, c, []))
)
