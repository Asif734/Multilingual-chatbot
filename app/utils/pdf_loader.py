# import fitz  # PyMuPDF

# def extract_text_from_pdf(pdf_path):
#     doc = fitz.open(pdf_path)
#     full_text = ""
#     for page in doc:
#         blocks = page.get_text("blocks")
#         blocks.sort(key=lambda b: (b[1], b[0]))  # Sort top-to-bottom, left-to-right
#         for b in blocks:
#             text = b[4]
#             full_text += text.strip() + "\n"
#     return full_text



import pdfplumber

def extract_text_from_pdf(pdf_path):
    full_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    return full_text