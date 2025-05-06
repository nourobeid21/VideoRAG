import streamlit as st
from retrieve import retrieve
from retrieve_pg import retrieve_pg

st.title("🎥 Video Q&A with RAG")

# 1) Let them choose the backend
backend = st.radio(
    "Choose retrieval backend:",
    ("FAISS (in-memory)", "pgvector (PostgreSQL)")
)

query = st.text_input("Ask a question about the video:")

if st.button("Search"):
    # 2) Dispatch to the right function
    if backend.startswith("FAISS"):
        answers = retrieve(query, top_k=3, use_semantic=True)
    else:
        answers = retrieve_pg(query, top_k=3)

    # 3) Display results
    if not answers:
        st.write("Sorry, I couldn't find an answer in the video.")
    else:
        for ans in answers:
            st.video("testvideo.mp4", start_time=ans["start"])
            st.write(f"Timestamp: {ans['start']}s — {ans['text'][:200]}…")
