from app.services.embedder import get_query_embedding
from app.core.config import VECTOR_DB_PATH
from chromadb import Client
from chromadb.config import Settings
import chromadb


# client = chromadb.Client()
# client =chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory=VECTOR_DB_PATH))
from chromadb import PersistentClient

client = PersistentClient(path=VECTOR_DB_PATH)
collection = client.get_or_create_collection(name="bangla_book")

def retrieve_relevant_chunks(query: str, k: int = 3):
    embedding = get_query_embedding(query)
    results = collection.query(query_embeddings=[embedding], n_results=k)
    return results["documents"][0]
