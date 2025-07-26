from sentence_transformers import SentenceTransformer
import numpy as np
import torch

embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def get_text_embedding(text: str) -> list[float]:
    """Generates a numerical embedding for a given text."""
    # Ensure the model is loaded once
    global embedding_model
    if embedding_model is None:
        embedding_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    embedding = embedding_model.encode(text)
    return embedding.tolist() # Convert numpy array to list for easier handling


# from sentence_transformers import SentenceTransformer

# embedding_model = SentenceTransformer("BAAI/bge-m3")

# def get_text_embedding(text: str) -> list[float]:
#     global embedding_model
#     if embedding_model is None:
#         embedding_model = SentenceTransformer("BAAI/bge-m3")

#     embedding = embedding_model.encode(text, normalize_embeddings=True)
#     return embedding.tolist()



# from transformers import AutoTokenizer, AutoModel
# import torch
# import torch.nn.functional as F

# # Load multilingual BGE model
# tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
# model = AutoModel.from_pretrained("BAAI/bge-m3")

# def get_text_embedding(text: str) -> list[float]:
#     try:
#         prompt = "Represent this sentence for searching relevant passages: "
#         full_text = prompt + text
#         print(f"Input text: {full_text}")

#         inputs = tokenizer(full_text, return_tensors='pt', truncation=True, padding=True)
#         print("Tokenized.")

#         with torch.no_grad():
#             outputs = model(**inputs)
#         print("Model inference done.")

#         attention_mask = inputs['attention_mask']
#         last_hidden = outputs.last_hidden_state
#         input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
#         sum_embeddings = torch.sum(last_hidden * input_mask_expanded, dim=1)
#         sum_mask = input_mask_expanded.sum(dim=1)
#         mean_pooled = sum_embeddings / sum_mask
#         print("Mean pooling done.")

#         normalized = F.normalize(mean_pooled, p=2, dim=1)
#         print("Normalized.")

#         return normalized[0].tolist()
    
#     except Exception as e:
#         print(f"ERROR in get_text_embedding: {e}")
#         raise