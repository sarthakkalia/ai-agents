"""
app.py  –  The Streamlit UI
============================
This is the front-end. It:
  - Lets the user upload a PDF or .txt file
  - Builds the RAG pipeline once (cached so it's not rebuilt on every message)
  - Shows a chat interface
  - Displays the "source chunks" used to generate each answer (transparency!)

Run with:  streamlit run app.py
"""
import os
import certifi

os.environ["SSL_CERT_FILE"] = certifi.where()

import os
import tempfile
import streamlit as st

# Import our RAG pipeline functions
from rag_pipeline import (
    load_and_split_documents,
    create_vector_store,
    build_rag_chain,
    ask,
)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.title("📄 RAG Chatbot – Chat with Your Document")
st.caption("Upload a PDF or .txt file, then ask questions about it.")



# ─────────────────────────────────────────────────────────────────────────────
# File upload
# ─────────────────────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload your document",
    type=["pdf", "txt"],
    help="Supported: PDF, plain text (.txt)",
)


# ─────────────────────────────────────────────────────────────────────────────
# Build (or load) the RAG chain
# st.cache_resource caches the chain in memory so it's only built ONCE
# even as the user keeps chatting. Without this, it re-embeds on every message!
# ─────────────────────────────────────────────────────────────────────────────

def get_rag_chain(file_bytes: bytes, file_name: str):
    """
    Builds the full RAG pipeline from the uploaded file.

    WHY @st.cache_resource?
      Streamlit reruns the entire script on every user action.
      Without caching, the document would be re-embedded on every keystroke.
      cache_resource persists the chain for the whole session.
    """
    # Save uploaded bytes to a temp file so LangChain loaders can read it
    suffix = ".pdf" if file_name.endswith(".pdf") else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Run the pipeline steps
    chunks       = load_and_split_documents(tmp_path)
    vector_store = create_vector_store(chunks)
    chain        = build_rag_chain(vector_store)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Main chat interface
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is not None:
    # Build or load the RAG chain for this file
    rag_chain = get_rag_chain(uploaded_file.read(), uploaded_file.name)

    # Chat interface
    st.subheader("💬 Ask questions about your document:")
    user_input = st.text_input("Your question:", placeholder="e.g. What are the main findings?")
    if user_input:
        with st.spinner("Generating answer…"):
            answer, source_chunks = ask(rag_chain, user_input)

        st.markdown(f"**Answer:** {answer}")
        with st.expander("Source chunks used for this answer (transparency!)"):
            for i, chunk in enumerate(source_chunks):
                st.markdown(f"**Chunk {i+1}:** {chunk.page_content}")
elif not uploaded_file:
    st.info("👆 Upload a PDF or .txt file above to begin.")