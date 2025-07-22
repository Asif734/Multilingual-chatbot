import re
import unicodedata

def clean_bangla_text(raw_text: str) -> str:
    # Remove common non-content patterns
    text = re.sub(r'Page\s*\d+', '', raw_text)
    text = re.sub(r'\s*[-=*_]{3,}\s*', '', text)

    # Normalize unicode
    text = unicodedata.normalize("NFKC", text)

    # Fix newline breaks
    text = re.sub(r'\n+', '\n', text)  # collapse multiple newlines
    text = re.sub(r'(?<!।)\n(?!\n)', ' ', text)  # join lines unless sentence ends

    # Remove excessive whitespace
    text = re.sub(r'\s{2,}', ' ', text)
    text = text.strip()

    # Remove English-only lines (optional, depends on document)
    text = '\n'.join([line for line in text.split('\n') if not re.match(r'^[a-zA-Z0-9\s,.!?;:\-]+$', line)])

    return text
