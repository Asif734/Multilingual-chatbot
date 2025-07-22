import sys
import os
import chromadb
from chromadb import Client
from chromadb.config import Settings
from app.utils.pdf_loader import extract_text_from_pdf
from app.utils.cleaner import clean_bangla_text
from app.utils.chunker import chunk_text
from app.services.embedder import get_chunk_embeddings


VECTOR_DB_PATH = "vectorstore/chroma_db1"
COLLECTION_NAME = "bangla_book"

# New way to create client with persist directory
settings = Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory=VECTOR_DB_PATH,
)

client = chromadb.Client()

collection = client.get_or_create_collection(name=COLLECTION_NAME)

# Load and process
pdf_path = r"C:\Users\Asif\VSCODE\Multilingual_AI_Assistant_RAG\app\data\HSC26-Bangla1st-Paper.pdf"
raw_text = extract_text_from_pdf(pdf_path)
cleaned_text = clean_bangla_text(raw_text)
print(cleaned_text)
chunks = chunk_text(cleaned_text)

embeddings = get_chunk_embeddings(chunks)

# Insert into vector DB
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=[f"chunk_{i}" for i in range(len(chunks))]
)


print(f"✅ Loaded {len(chunks)} chunks into vector DB.")
