import streamlit as st
from retrieve import retrieve
from retrieve_pg import retrieve_pg

st.title("🎥 Video Q&A with RAG")

# 1) Retrieval type
mode = st.radio(
    "Mode:",
    ("Semantic", "Lexical")
)

# 2) If semantic, pick backend
semantic_backend = None
if mode == "Semantic":
    semantic_backend = st.radio(
        "Semantic backend:",
        ("FAISS (in-memory)", "pgvector (PostgreSQL)")
    )

# 3) If pgvector, pick index
pg_index = None
if semantic_backend and semantic_backend.startswith("pgvector"):
    pg_index = st.radio("pgvector index:", ("hnsw", "ivfflat"))

# 4) If lexical, pick method
lexical_method = None
if mode == "Lexical":
    lexical_method = st.radio(
        "Lexical method:",
        ("tfidf", "bm25")
    )

query = st.text_input("Ask a question about the video:")

if st.button("Search"):
    # Dispatch
    if mode == "Semantic":
        if semantic_backend.startswith("FAISS"):
            answers = retrieve(query, top_k=3, semantic=True, threshold=0.3)
        else:
            answers = retrieve_pg(query, top_k=3, index_type=pg_index, threshold=0.3)
    else:  # Lexical
        answers = retrieve(query, top_k=3, lexical=True, lexical_method=lexical_method, threshold=0.3)

    # Display
    if not answers:
        st.write("Sorry, I couldn't find an answer in the video.")
    else:
        for ans in answers:
            st.video("testvideo.mp4", start_time=ans["start"])
            st.write(f"Timestamp: {ans['start']}s — {ans['text'][:200]}…")
