import re

def clean_text(text: str) -> str:
    # Replace multiple spaces/newlines/tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()