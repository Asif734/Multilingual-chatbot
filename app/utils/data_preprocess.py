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
    print(text)
    return text


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





