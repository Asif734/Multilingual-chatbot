from sentence_transformers import SentenceTransformer
import numpy as np

# Load a multilingual embedding model
# 'paraphrase-multilingual-MiniLM-L12-v2' is a good choice for many languages, including English and Bengali.
# You might experiment with other models like 'LaBSE' for better multilingual performance if needed.
embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_text_embedding(text: str) -> list[float]:
    """Generates a numerical embedding for a given text."""
    # Ensure the model is loaded once
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    embedding = embedding_model.encode(text)
    return embedding.tolist() # Convert numpy array to list for easier handling

