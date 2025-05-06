from sentence_transformers import SentenceTransformer
import json

model = SentenceTransformer("all-MiniLM-L6-v2")
with open("data/transcripts/chunks.jsonl") as f:
    chunks = [json.loads(l) for l in f]

texts = [c["text"] for c in chunks]
text_embs = model.encode(texts, show_progress_bar=True)

# save to disk
import numpy as np
np.save("data/embeddings/text_embs.npy", text_embs)
