from sentence_transformers import SentenceTransformer
from app.core.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

def get_query_embedding(text: str):
    return model.encode([text])[0]


def get_chunk_embeddings(chunks: list[str]):
    return model.encode(chunks)