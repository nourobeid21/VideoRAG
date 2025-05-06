# build_indexes.py
import faiss
import numpy as np
import json, os

# Load text embeddings
text_embs = np.load("data/embeddings/text_embs.npy")
d_text   = text_embs.shape[1]
# Normalize for cosine similarity
faiss.normalize_L2(text_embs)

text_index = faiss.IndexFlatIP(d_text)       # inner-product on L2-normalized = cosine
text_index.add(text_embs)
faiss.write_index(text_index, "data/indexes/text_faiss.index")

# (Optional) do the same for image embeddings
img_embs = np.load("data/embeddings/img_embs.npy")
d_img    = img_embs.shape[1]
faiss.normalize_L2(img_embs)
img_index = faiss.IndexFlatIP(d_img)
img_index.add(img_embs)
faiss.write_index(img_index, "data/indexes/img_faiss.index")
