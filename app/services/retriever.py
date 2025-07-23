from typing import List
from app.services.embedding import get_text_embedding
from app.scripts.vector_store import InMemoryVectorStore
import os

# Global instance of the vector store (for simplicity in this example)
# In a production FastAPI app, you might inject this as a dependency.
rag_vector_store = InMemoryVectorStore()

def initialize_retriever(pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, index_path: str = "rag_index"):
    """
    Initializes the RAG retriever by processing the PDF, generating embeddings,
    and building/loading the vector store.
    """
    from app.utils.data_preprocess import extract_text_from_pdf, chunk_text

    print("Initializing RAG retriever...")
    # Attempt to load existing index first
    rag_vector_store.load_index(index_path)

    if not rag_vector_store.is_built or not rag_vector_store.documents:
        print("Index not found or empty. Building new index from PDF...")
        try:
            full_text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"Generated {len(chunks)} chunks from PDF.")

            embeddings = []
            print("Generating embeddings for chunks (this might take a while for large PDFs)...")
            for i, chunk in enumerate(chunks):
                embeddings.append(get_text_embedding(chunk))
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1} embeddings...")

            rag_vector_store.add_documents(chunks, embeddings)
            rag_vector_store.build_index()
            rag_vector_store.save_index(index_path)
            print("RAG retriever initialized and index saved.")
        except FileNotFoundError:
            print(f"Error: PDF file not found at {pdf_path}. Please ensure it exists.")
            # Handle this gracefully, maybe by raising an exception or returning a status
            raise
        except Exception as e:
            print(f"An error occurred during retriever initialization: {e}")
            raise
    else:
        print("RAG retriever initialized from loaded index.")


def retrieve_relevant_chunks(query: str, k: int = 3) -> List[str]:
    """
    Retrieves the most relevant text chunks from the knowledge base for a given query.
    """
    if not rag_vector_store.is_built:
        print("Warning: Retriever not initialized. Please call initialize_retriever() first.")
        return []

    query_embedding = get_text_embedding(query)
    results = rag_vector_store.search(query_embedding, k=k)

    # Return just the text content of the retrieved documents
    relevant_texts = [doc_text for doc_text, _ in results]
    return relevant_texts

if __name__ == "__main__":
    # Test the retrieval module
    pdf_file_name = r"C:\Users\Asif\VSCODE\Multilingual_AI_Assistant_RAG\app\data\HSC26-Bangla1st-Paper.pdf" # Ensure this file is in the same directory

    # Initialize the retriever (this would typically run once when the app starts)
    try:
        initialize_retriever(pdf_file_name)

        print("\nTesting retrieval with an English query:")
        english_query = "What is the main topic of this paper?"
        retrieved_en = retrieve_relevant_chunks(english_query, k=2)
        print(f"Relevant chunks for '{english_query}':")
        for i, chunk in enumerate(retrieved_en):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")

        print("\nTesting retrieval with a Bengali query:")
        bengali_query = "এই প্রবন্ধের মূল বিষয় কী?"
        retrieved_bn = retrieve_relevant_chunks(bengali_query, k=2)
        print(f"Relevant chunks for '{bengali_query}':")
        for i, chunk in enumerate(retrieved_bn):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")

    except Exception as e:
        print(f"An error occurred during retrieval test: {e}")