# import re

# def clean_text(text: str) -> str:
#     # Replace multiple spaces/newlines/tabs with a single space
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()
import torch

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU ")

from pdf2image import convert_from_path
import easyocr
import os
from PIL import Image # Keep PIL for potential image processing if needed
import numpy as np # Import numpy for array conversion

# Set path to your Poppler bin (change if needed). This is for pdf2image.
POPPLER_PATH = r"C:\Users\Asif\VSCODE\poppler-24.02.0\Library\bin"

# Initialize EasyOCR reader once.
# Note: You're getting "Neither CUDA nor MPS are available - defaulting to CPU."
# This means easyocr is running on CPU. If you have an NVIDIA GPU,
# ensure you've installed the CUDA-enabled version of PyTorch for speed.
# For now, running on CPU is fine, but expect it to be slower for many pages.
try:
    reader = easyocr.Reader(['bn', 'en'], gpu=True) # Keep gpu=True if you *intend* to use GPU, it will fallback to CPU if not found
    print("EasyOCR reader initialized with Bengali and English language models.")
except Exception as e:
    print(f"Error initializing EasyOCR reader: {e}")
    print("Please ensure EasyOCR and PyTorch are correctly installed. Check GPU settings if 'gpu=True'.")
    exit()

def extract_text_from_pdf_with_easyocr(pdf_path: str, dpi: int = 300) -> str:
    """
    Extracts text from a PDF using EasyOCR.

    Args:
        pdf_path (str): The path to the input PDF file.
        dpi (int): The DPI (dots per inch) to render PDF pages as images.
                   Higher DPI can improve OCR accuracy but uses more memory and time.

    Returns:
        str: The full extracted text from the PDF.
    """
    if not os.path.exists(pdf_path):
        return f"Error: PDF file not found at '{pdf_path}'"

    full_text = ""
    try:
        # Convert PDF pages to PIL Image objects
        pages = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=dpi)
        print(f"Converted {len(pages)} pages from '{os.path.basename(pdf_path)}'.")

        for i, page in enumerate(pages):
            print(f"Processing page {i+1}/{len(pages)} with EasyOCR...")

            # --- CRITICAL CHANGE HERE ---
            # Convert the PIL Image object to a NumPy array.
            # EasyOCR reliably works with NumPy arrays.
            image_np = np.array(page)

            # Perform OCR on the image. detail=0 returns only the text.
            results = reader.readtext(image_np, detail=0) # Pass the NumPy array

            # Join the detected text lines for the current page
            page_text = "\n".join(results)
            full_text += f"\n--- Page {i+1} ---\n{page_text}"

    except Exception as e:
        full_text += f"\nAn error occurred during PDF conversion or OCR for '{pdf_path}': {e}"
        full_text += "\nPlease ensure Poppler is installed and POPPLER_PATH is correctly set."
        full_text += "\nAlso check if EasyOCR models are downloaded and if GPU setup is correct (if using GPU)."

    return full_text

# === Run the extraction ===
if __name__ == "__main__":
    # Define the path to your PDF file
    pdf_file = r"C:\Users\Asif\VSCODE\Multilingual_AI_Assistant_RAG\app\data\HSC26-Bangla1st-Paper.pdf"

    # Extract text using the EasyOCR function
    text = extract_text_from_pdf_with_easyocr(pdf_file, dpi=300) # You can adjust DPI here

    # Print the first 1000 characters of the extracted text
    print("\n--- Extracted Text (First 1000 characters) ---")
    print(text[:1000])

    # Optionally, save the full extracted text to a file
    output_txt_file = "extracted_text_from_HSC26_Bangla1st-Paper.txt"
    try:
        with open(output_txt_file, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"\nFull extracted text saved to '{output_txt_file}'")
    except Exception as e:
        print(f"\nError saving extracted text to file: {e}")