import fitz # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import re # For regular expressions for cleaning

def clean_text(text: str) -> str:
    """
    Performs basic text cleaning:
    - Removes excessive whitespace (multiple spaces, tabs, newlines)
    - Normalizes newlines
    - Strips leading/trailing whitespace
    """
    if not text:
        return ""
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces/tabs with a single space
    text = re.sub(r'[ \t]+', ' ', text)
    # Remove leading/trailing whitespace from each line
    text = "\n".join([line.strip() for line in text.split('\n')])
    # Remove any leading/trailing whitespace from the whole text
    text = text.strip()
   
    return text

# def extract_text_from_pdf(pdf_path: str) -> str:
#     """Extracts all text from a PDF file using PyMuPDF (fitz) and cleans it."""
#     raw_text = ""
#     try:
#         doc = fitz.open(pdf_path)
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             page_text = page.get_text("text")
#             if page_text:
#                 raw_text += page_text + "\n" # Add a newline between pages
#         doc.close()
#         print(f"Successfully extracted raw text from {pdf_path} using PyMuPDF (fitz).")

#         if not raw_text.strip():
#             print(f"Warning: PyMuPDF extracted empty or very little raw text from {pdf_path}. This might be a scanned PDF.")
#             # If it's a scanned PDF, OCR (e.g., pytesseract) would be the next step here.
#             # For this project, we'll assume fitz handles the primary text extraction.

#     except Exception as e:
#         print(f"Error extracting text from PDF with PyMuPDF (fitz): {e}")
#         raise

#     # Apply cleaning after extraction
#     cleaned_text = clean_text(raw_text)
#     print(f"Text cleaned. Original chars: {len(raw_text)}, Cleaned chars: {len(cleaned_text)}")
    
#     if not cleaned_text.strip():
#         raise ValueError("No meaningful text could be extracted and cleaned from the PDF.")

#     return cleaned_text

def chunk_text(text: str, chunk_size: int = 2000, chunk_overlap: int = 400) -> list[str]:
    """Splits text into smaller, overlapping chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_text(text)
    return chunks

if __name__ == "__main_": # Still altered to prevent automatic run during import
    pdf_file_name = "HSC26-Bangla1st-Paper.pdf"

    print(f"Attempting to extract and clean text from {pdf_file_name} using PyMuPDF (fitz)...")
    try:
        full_text = extract_text_from_pdf(pdf_file_name)
        print(f"Text extracted and cleaned. Total characters: {len(full_text)}")

        chunks = chunk_text(full_text)
        print(f"Text chunked. Total chunks: {len(chunks)}")
        print("\nFirst 3 chunks (check for readability and correct characters):")
        for i, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {i+1} ---\n{chunk}\n")
    except FileNotFoundError:
        print(f"Error: The file '{pdf_file_name}' was not found. Please ensure it's in the correct directory.")
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")



