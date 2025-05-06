# load_into_pg.py
import psycopg2, json, numpy as np

conn = psycopg2.connect(dbname="video_rag", user="postgres", password="nina", host="localhost")
cur  = conn.cursor()

# Load chunk metadata
chunks = [json.loads(l) for l in open("data/transcripts/chunks.jsonl")]

# Load your saved embeddings
text_embs = np.load("data/embeddings/text_embs.npy")
img_embs  = np.load("data/embeddings/img_embs.npy")

for c, t_emb, v_emb in zip(chunks, text_embs, img_embs):
    cur.execute(
      "INSERT INTO chunks (start_s,end_s,text,text_emb,img_emb) VALUES (%s,%s,%s,%s,%s)",
      (c["start"], c["end"], c["text"], t_emb.tolist(), v_emb.tolist())
    )

conn.commit()
cur.close()
conn.close()
