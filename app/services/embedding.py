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

if __name__ == "__main__":
    # Test the embedding generator
    sample_text_en = "Hello, how are you?"
    sample_text_bn = "কেমন আছেন আপনি?"
    sample_text_related_en = "How's your day going?"

    print(f"Embedding for English text '{sample_text_en}':")
    emb_en = get_text_embedding(sample_text_en)
    print(f"Length of embedding: {len(emb_en)}")
    print(f"First 5 elements: {emb_en[:5]}\n")

    print(f"Embedding for Bengali text '{sample_text_bn}':")
    emb_bn = get_text_embedding(sample_text_bn)
    print(f"Length of embedding: {len(emb_bn)}")
    print(f"First 5 elements: {emb_bn[:5]}\n")

    print(f"Embedding for related English text '{sample_text_related_en}':")
    emb_related_en = get_text_embedding(sample_text_related_en)
    print(f"Length of embedding: {len(emb_related_en)}")
    print(f"First 5 elements: {emb_related_en[:5]}\n")

    # You can also compute similarity between embeddings for testing
    from scipy.spatial.distance import cosine
    similarity_en_bn = 1 - cosine(emb_en, emb_bn)
    similarity_en_related_en = 1 - cosine(emb_en, emb_related_en)
    print(f"Similarity between English and Bengali: {similarity_en_bn}")
    print(f"Similarity between two related English sentences: {similarity_en_related_en}")