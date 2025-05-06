# build_lexical.py
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
import json, pickle

# Load your chunks
chunks = [json.loads(l) for l in open("data/transcripts/chunks.jsonl")]

# Extract texts
texts = [c["text"] for c in chunks]

# TF-IDF
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(texts)
with open("data/indexes/tfidf_vec.pkl","wb") as f:
    pickle.dump(tfidf, f)

# BM25
tokenized = [t.split() for t in texts]
bm25 = BM25Okapi(tokenized)
with open("data/indexes/bm25_obj.pkl","wb") as f:
    pickle.dump(bm25, f)



# retrieve.py
import faiss, numpy as np, pickle, json
from sentence_transformers import SentenceTransformer

# Load models & indexes once
st_model   = SentenceTransformer("all-MiniLM-L6-v2")
text_index = faiss.read_index("data/indexes/text_faiss.index")
img_index  = faiss.read_index("data/indexes/img_faiss.index")  # if using

tfidf      = pickle.load(open("data/indexes/tfidf_vec.pkl","rb"))
bm25       = pickle.load(open("data/indexes/bm25_obj.pkl","rb"))
chunks     = [json.loads(l) for l in open("data/transcripts/chunks.jsonl")]

def retrieve(query, top_k=5, use_semantic=True, use_lexical=False, threshold=0.2):
    results = []
    
    # --- Semantic retrieval
    if use_semantic:
        q_emb = st_model.encode([query], normalize_embeddings=True)
        D, I = text_index.search(q_emb, top_k)  # D: scores, I: indices
        for score, idx in zip(D[0], I[0]):
            if score < threshold: break
            c = chunks[idx]
            results.append({"chunk_id": idx,
                            "start": c["start"],
                            "end": c["end"],
                            "text":  c["text"],
                            "score": float(score),
                            "method":"semantic"})
    
    # --- Lexical retrieval (TF-IDF + cosine)
    if use_lexical:
        q_vec = tfidf.transform([query])
        sims  = (q_vec @ tfidf_matrix.T).toarray()[0]
        lex_idxs = np.argsort(-sims)[:top_k]
        for idx in lex_idxs:
            results.append({"chunk_id": idx,
                            "start": chunks[idx]["start"],
                            "end":   chunks[idx]["end"],
                            "text":  chunks[idx]["text"],
                            "score": float(sims[idx]),
                            "method":"tfidf"})
        # BM25 similarly via bm25.get_scores(query.split())
    
    # Sort by score descending
    results = sorted(results, key=lambda x: -x["score"])
    
    return results
