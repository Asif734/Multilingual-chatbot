# rag_evaluation.py
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity

def evaluate_groundedness(answer: str, retrieved_chunks: List[str], expected: str) -> bool:
    """
    Check if the expected answer is contained in any retrieved chunk.
    """
    return any(expected in chunk for chunk in retrieved_chunks)

def evaluate_relevance(query_embedding, retrieved_embeddings) -> float:
    """
    Calculate average cosine similarity between the query and top-k retrieved chunks.
    """
    scores = cosine_similarity([query_embedding], retrieved_embeddings)
    return float(scores.mean())

def run_sample_tests(
    test_cases: List[Tuple[str, str]], 
    retriever_func, 
    answer_generator_func
):
    """
    Run multiple test cases and print evaluation results.
    Each test_case is a (question, expected_answer) tuple.
    """
    for question, expected in test_cases:
        chunks = retriever_func(question)
        answer = answer_generator_func(question, chunks)
        
        is_grounded = evaluate_groundedness(answer, chunks, expected)
        print(f"\nQuestion: {question}")
        print(f"Generated Answer: {answer}")
        print(f"Expected: {expected}")
        print(f"✅ Grounded: {is_grounded}")
