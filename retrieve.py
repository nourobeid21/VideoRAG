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
    :param threshold: minimum score to include result
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
            # filter by threshold and take top_k
            candidates = [(idx, sims[idx]) for idx in np.argsort(-sims)]
            count = 0
            for idx, score in candidates:
                if score < threshold or count >= top_k:
                    break
                c = chunks[idx]
                results.append({
                    "chunk_id": idx,
                    "start":    c["start"],
                    "end":      c["end"],
                    "text":     c["text"],
                    "score":    float(score),
                    "method":   "tfidf"
                })
                count += 1
        else:  # bm25
            tokenized = query.split()
            scores    = bm25.get_scores(tokenized)
            candidates = [(idx, scores[idx]) for idx in np.argsort(-scores)]
            count = 0
            for idx, score in candidates:
                if score < threshold or count >= top_k:
                    break
                c = chunks[idx]
                results.append({
                    "chunk_id": idx,
                    "start":    c["start"],
                    "end":      c["end"],
                    "text":     c["text"],
                    "score":    float(score),
                    "method":   "bm25"
                })
                count += 1

    # sort by score
    results = sorted(results, key=lambda x: -x["score"])
    return results
