# retrieve.py
import faiss, numpy as np, pickle, json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Load models & indexes once
st_model   = SentenceTransformer("all-MiniLM-L6-v2")
text_index = faiss.read_index("data/indexes/text_faiss.index")

# Load lexical artifacts
with open("data/indexes/tfidf_vec.pkl", "rb") as f:
    tfidf = pickle.load(f)
# We need the matrix too
from sklearn.feature_extraction.text import TfidfVectorizer
texts = [json.loads(l)["text"] for l in open("data/transcripts/chunks.jsonl")]
tfidf_matrix = tfidf.transform(texts)

with open("data/indexes/bm25_obj.pkl", "rb") as f:
    bm25 = pickle.load(f)

chunks = [json.loads(l) for l in open("data/transcripts/chunks.jsonl")]

def retrieve(
    query,
    top_k=5,
    semantic=False,
    lexical=False,
    lexical_method="tfidf",
    threshold=0.2
):
    """
    :param semantic: use FAISS semantic search
    :param lexical: use lexical search
    :param lexical_method: "tfidf" or "bm25"
    """
    results = []

    # Semantic
    if semantic:
        q_emb = st_model.encode([query], normalize_embeddings=True)
        D, I = text_index.search(q_emb, top_k)
        for score, idx in zip(D[0], I[0]):
            if score < threshold:
                break
            c = chunks[idx]
            results.append({
                "chunk_id": idx,
                "start":    c["start"],
                "end":      c["end"],
                "text":     c["text"],
                "score":    float(score),
                "method":   "semantic"
            })

    # Lexical
    if lexical:
        if lexical_method == "tfidf":
            q_vec = tfidf.transform([query])
            sims  = (q_vec @ tfidf_matrix.T).toarray()[0]
            idxs  = np.argsort(-sims)[:top_k]
            for idx in idxs:
                results.append({
                    "chunk_id": idx,
                    "start":    chunks[idx]["start"],
                    "end":      chunks[idx]["end"],
                    "text":     chunks[idx]["text"],
                    "score":    float(sims[idx]),
                    "method":   "tfidf"
                })
        else:  # bm25
            tokenized = query.split()
            scores    = bm25.get_scores(tokenized)
            idxs      = np.argsort(-scores)[:top_k]
            for idx in idxs:
                results.append({
                    "chunk_id": idx,
                    "start":    chunks[idx]["start"],
                    "end":      chunks[idx]["end"],
                    "text":     chunks[idx]["text"],
                    "score":    float(scores[idx]),
                    "method":   "bm25"
                })

    # sort by score
    results = sorted(results, key=lambda x: -x["score"])
    return results
