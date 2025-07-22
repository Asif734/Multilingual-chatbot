import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
VECTOR_DB_PATH = "vectorstore/chroma_db"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
