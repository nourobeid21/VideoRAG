# retrieve_pg.py
import psycopg2
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
conn  = psycopg2.connect(
    dbname="video_rag",
    user="postgres",
    password="nina",
    host="localhost"
)

def retrieve_pg(query, top_k=5):
    # 1) Embed the query
    q_emb = model.encode([query], normalize_embeddings=True)[0].tolist()

    # 2) Query pgvector
    with conn.cursor() as cur:
        cur.execute("""
          SET pgvector.hnsw.ef_search = 50;
          SELECT chunk_id, start_s, end_s, text,
                 1 - (text_emb <=> %s::vector) AS score
          FROM chunks
          ORDER BY text_emb <=> %s::vector
          LIMIT %s;
        """, (q_emb, q_emb, top_k))
        rows = cur.fetchall()

    # 3) Pack into dicts
    results = []
    for chunk_id, start_s, end_s, text, score in rows:
        results.append({
            "chunk_id": chunk_id,
            "start":     start_s,
            "end":       end_s,
            "text":      text,
            "score":     float(score),
            "method":    "pgvector"
        })
    return results
