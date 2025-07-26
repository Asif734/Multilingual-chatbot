import faiss
import numpy as np
from typing import List, Tuple
import os
import pickle

class InMemoryVectorStore:
    def __init__(self):
        self.index = None
        self.documents = [] # Stores the original text chunks
        self.chunk_embeddings = [] # Stores the embeddings of the chunks
        self.is_built = False

    def add_documents(self, chunks: List[str], embeddings: List[List[float]]):
        """Adds text chunks and their embeddings to the store."""
        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks and embeddings must be equal.")

        self.documents.extend(chunks)
        self.chunk_embeddings.extend(embeddings)
        self.is_built = False # Mark for rebuilding the FAISS index

    def build_index(self):
        """Builds the FAISS index from the stored embeddings."""
        if not self.chunk_embeddings:
            print("No embeddings to build index from.")
            return

        # Convert list of lists to a 2D numpy array
        embeddings_np = np.array(self.chunk_embeddings).astype('float32')

        # Get embedding dimension
        dimension = embeddings_np.shape[1]

        # Initialize a FAISS index (e.g., IndexFlatL2 for L2 distance)
        # For small scale, IndexFlatL2 is fine. For larger scales, consider HNSW, IVF, etc.
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        self.is_built = True
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def search(self, query_embedding: List[float], k: int = 3) -> List[Tuple[str, float]]:
        """
        Searches the vector store for the top-k most similar documents.

        Args:
            query_embedding: The embedding of the query.
            k: The number of top relevant documents to retrieve.

        Returns:
            A list of tuples, where each tuple contains (document_text, similarity_score).
        """
        if not self.is_built or self.index is None:
            raise RuntimeError("FAISS index has not been built. Call build_index() first.")
        if not self.documents:
            return [] # No documents to search

        query_embedding_np = np.array([query_embedding]).astype('float32')

        # D: distances, I: indices of the nearest neighbors
        distances, indices = self.index.search(query_embedding_np, k)

        results = []
        # FAISS returns distances, we often want similarity (e.g., 1 - distance if normalized)
        # Here, lower L2 distance means higher similarity.
        for i, idx in enumerate(indices[0]):
            if idx < len(self.documents): # Ensure index is valid
                results.append((self.documents[idx], distances[0][i]))
        return results

    def save_index(self, path: str):
        """Saves the FAISS index and documents to disk."""
        if self.index is None:
            print("No index to save.")
            return

        # Save FAISS index
        faiss.write_index(self.index, f"{path}.faiss")
        # Save documents and embeddings
        with open(f"{path}.pkl", 'wb') as f:
            pickle.dump({"documents": self.documents, "chunk_embeddings": self.chunk_embeddings}, f)
        print(f"Index and documents saved to {path}.faiss and {path}.pkl")

    def load_index(self, path: str):
        """Loads the FAISS index and documents from disk."""
        if os.path.exists(f"{path}.faiss") and os.path.exists(f"{path}.pkl"):
            self.index = faiss.read_index(f"{path}.faiss")
            with open(f"{path}.pkl", 'rb') as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.chunk_embeddings = data["chunk_embeddings"]
            self.is_built = True
            print(f"Index and documents loaded from {path}.faiss and {path}.pkl")
        else:
            print(f"Warning: Files for loading index not found at {path}.faiss or {path}.pkl. Starting fresh.")


