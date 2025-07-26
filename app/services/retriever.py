from typing import List
from app.services.embedding import get_text_embedding
from app.scripts.vector_store import InMemoryVectorStore
from app.utils.data_preprocess import chunk_text

import os

# Global instance of the vector store (used by main.py and elsewhere)
rag_vector_store = InMemoryVectorStore()


def initialize_retriever_from_text(text_path: str, chunk_size: int = 1000, chunk_overlap: int = 200, index_path: str = "rag_index"):
    """
    Initializes the RAG retriever from a plain text file (already extracted),
    chunks it, embeds the chunks, and stores the embeddings in a vector index.
    """
    print("Initializing RAG retriever from text...")

    # Load existing index if available
    rag_vector_store.load_index(index_path)

    if not rag_vector_store.is_built or not rag_vector_store.documents:
        print("Index not found or empty. Building new index from text...")

        try:
            with open(text_path, "r", encoding="utf-8") as file:
                full_text = file.read()

            chunks = chunk_text(full_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            print(f"✅ Generated {len(chunks)} text chunks.")

            embeddings = []
            print("⏳ Generating embeddings for each chunk...")
            for i, chunk in enumerate(chunks):
                embeddings.append(get_text_embedding(chunk))
                if (i + 1) % 100 == 0:
                    print(f"Processed {i+1} embeddings...")

            rag_vector_store.add_documents(chunks, embeddings)
            rag_vector_store.build_index()
            rag_vector_store.save_index(index_path)

            print("✅ RAG retriever initialized and index saved.")

        except FileNotFoundError:
            print(f"❌ Error: Text file not found at {text_path}.")
            raise
        except Exception as e:
            print(f"❌ Error during retriever initialization: {e}")
            raise
    else:
        print("✅ RAG retriever loaded from saved index.")


def retrieve_relevant_chunks(query: str, k: int = 3) -> List[str]:
    """
    Retrieves the most relevant text chunks from the knowledge base for a given query.
    """
    if not rag_vector_store.is_built:
        print("⚠️ Warning: Retriever not initialized. Please call `initialize_retriever_from_text()` first.")
        return []

    query_embedding = get_text_embedding(query)
    results = rag_vector_store.search(query_embedding, k=k)

    relevant_texts = [doc_text for doc_text, _ in results]
    return relevant_texts


# # Optional: CLI Test
# if __name__ == "__main__":
#     text_path = r"app/data/extracted_text_from_HSC26_Bangla1st-Paper.txt"

#     try:
#         initialize_retriever_from_text(text_path)

#         # Test English query
#         english_query = "What is the main topic of this paper?"
#         retrieved_en = retrieve_relevant_chunks(english_query, k=2)
#         print(f"\n🔍 English Query: {english_query}")
#         for i, chunk in enumerate(retrieved_en):
#             print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

#         # Test Bengali query
#         bengali_query = "এই প্রবন্ধের মূল বিষয় কী?"
#         retrieved_bn = retrieve_relevant_chunks(bengali_query, k=2)
#         print(f"\n🔍 Bengali Query: {bengali_query}")
#         for i, chunk in enumerate(retrieved_bn):
#             print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

#     except Exception as e:
#         print(f"❌ Retrieval test failed: {e}")
