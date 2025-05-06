import psycopg2
from sentence_transformers import SentenceTransformer

# Load the sentence‐transformer model once
model = SentenceTransformer("all-MiniLM-L6-v2")

# Persistent DB connection
conn = psycopg2.connect(
    dbname="video_rag",
    user="postgres",
    password="nina",
    host="localhost"
)

def retrieve_pg(query, top_k=5, index_type="hnsw", threshold=0.2):
    """
    Retrieve top_k chunks from Postgres pgvector using either HNSW or IVFFLAT,
    applying a score threshold filter.
    
    :param query: the user question
    :param top_k: number of results
    :param index_type: "hnsw" or "ivfflat"
    :param threshold: minimum similarity score to include
    :returns: list of dicts with keys chunk_id, start, end, text, score, method
    """
    # 1) Embed the query
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    with conn.cursor() as cur:
        # 2) Configure the index type
        if index_type.lower() == "hnsw":
            cur.execute("SET pgvector.hnsw.ef_search = 50;")
            cur.execute("RESET pgvector.ivfflat.probes;")
        else:
            cur.execute("SET pgvector.ivfflat.probes = 10;")
            cur.execute("RESET pgvector.hnsw.ef_search;")
        
        # 3) Run the similarity query
        cur.execute("""
          SELECT chunk_id, start_s, end_s, text,
                 1 - (text_emb <=> %s::vector) AS score
          FROM chunks
          ORDER BY text_emb <=> %s::vector
          LIMIT %s;
        """, (q_emb, q_emb, top_k))
        rows = cur.fetchall()

    # 4) Pack into dicts, apply threshold and limit
    results = []
    count = 0
    for chunk_id, start_s, end_s, text, score in rows:
        if score < threshold or count >= top_k:
            break
        results.append({
            "chunk_id": chunk_id,
            "start":    start_s,
            "end":      end_s,
            "text":     text,
            "score":    float(score),
            "method":   f"pgvector-{index_type.lower()}"
        })
        count += 1
    return results
