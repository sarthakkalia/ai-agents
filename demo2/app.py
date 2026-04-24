"""
app.py  ─  Streamlit Chat UI
=============================
Renders the structured MedicalAnswer response beautifully:
  - Main answer in a clear block
  - Confidence badge (color-coded)
  - Key facts as expandable cards
  - Follow-up question buttons (click to ask them!)
  - Source document tags
  - Medical disclaimer footer

Run:  streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv

from rag_chain import build_retriever, build_rag_chain, query

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Medical RAG Chatbot",
    page_icon="🩺",
    layout="wide",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — clean, medical-themed UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .confidence-high   { background:#d4edda; color:#155724; padding:4px 12px;
                         border-radius:12px; font-weight:600; font-size:13px; }
    .confidence-medium { background:#fff3cd; color:#856404; padding:4px 12px;
                         border-radius:12px; font-weight:600; font-size:13px; }
    .confidence-low    { background:#f8d7da; color:#721c24; padding:4px 12px;
                         border-radius:12px; font-weight:600; font-size:13px; }
    .fact-card { background:#f8f9fa; border-left:4px solid #0d6efd;
                 padding:10px 14px; margin:6px 0; border-radius:4px; }
    .topic-tag { background:#e7f1ff; color:#0d6efd; padding:2px 8px;
                 border-radius:10px; font-size:11px; font-weight:600; }
    .disclaimer { background:#fff8e1; border:1px solid #ffe082;
                  padding:10px; border-radius:6px; font-size:13px; color:#5d4037; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — API keys and info
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    groq_key = st.text_input("Groq API Key", type="password",
                              value=os.getenv("GROQ_API_KEY", ""),
                              help="Free at console.groq.com")
    pine_key = st.text_input("Pinecone API Key", type="password",
                              value=os.getenv("PINECONE_API_KEY", ""),
                              help="Free at app.pinecone.io")
    index_name = st.text_input("Pinecone Index Name",
                                value=os.getenv("PINECONE_INDEX_NAME", "medical-rag"))

    if groq_key:   os.environ["GROQ_API_KEY"]       = groq_key
    if pine_key:   os.environ["PINECONE_API_KEY"]    = pine_key
    if index_name: os.environ["PINECONE_INDEX_NAME"] = index_name

    st.markdown("---")
    st.markdown("### 📚 Loaded Documents")
    st.markdown("""
    - `diabetes.pdf`
    - `renal_failure.pdf`
    - `A-new-threat-to-obesity--vanity-sizing.pdf`
    - *(any PDF you added to pdfs/)*
    """)

    st.markdown("---")
    st.markdown("### 🔬 How it works")
    st.markdown("""
    1. Your question → HuggingFace embedding
    2. Pinecone finds top-5 similar chunks
    3. Groq LLaMA 3.3 70B answers
    4. Pydantic validates the output schema
    5. You get a structured, verified answer
    """)

    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Build (and cache) the RAG chain
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🔗 Connecting to Pinecone and loading models…")
def get_chain():
    """
    Cached so it only builds once per session.
    Subsequent messages reuse the same retriever + chain.
    """
    retriever = build_retriever(k=5)
    chain     = build_rag_chain(retriever)
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Helper: render one structured answer
# ─────────────────────────────────────────────────────────────────────────────
def render_answer(data: dict):
    """Renders a MedicalAnswer.to_display() dict as rich Streamlit UI."""

    # --- Confidence badge ---
    conf  = data["confidence"]
    badge = f'<span class="confidence-{conf}">Confidence: {conf.upper()}</span>'
    st.markdown(badge, unsafe_allow_html=True)
    st.write("")

    # --- Main answer ---
    st.markdown(data["answer"])

    # --- Conditions mentioned ---
    if data["conditions"]:
        st.markdown("**🏥 Conditions referenced:** " +
                    " · ".join(f"`{c}`" for c in data["conditions"]))

    # --- Key facts ---
    if data["key_facts"]:
        st.markdown("**📋 Key facts from the documents:**")
        for kf in data["key_facts"]:
            st.markdown(
                f'<div class="fact-card">'
                f'<span class="topic-tag">{kf["topic"]}</span>&nbsp;&nbsp;'
                f'{kf["fact"]}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # --- Follow-up questions as clickable buttons ---
    if data["follow_up"]:
        st.markdown("**💡 You might also want to ask:**")
        cols = st.columns(len(data["follow_up"]))
        for i, q in enumerate(data["follow_up"]):
            if cols[i].button(q, key=f"followup_{q[:30]}"):
                # Inject the question as the next user message
                st.session_state.pending_question = q
                st.rerun()

    # --- Disclaimer ---
    st.markdown(
        f'<div class="disclaimer">⚠️ {data["disclaimer"]}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main chat interface
# ─────────────────────────────────────────────────────────────────────────────
st.title("🩺 Medical Document RAG Chatbot")
st.caption("Ask questions about Diabetes, Renal Failure, and Obesity. "
           "Powered by Groq LLaMA 3.3 · Pinecone · Pydantic structured output.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []      # list of {role, content, data}
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None

keys_ready = os.getenv("GROQ_API_KEY") and os.getenv("PINECONE_API_KEY")

if not keys_ready:
    st.warning("👈 Enter your Groq and Pinecone API keys in the sidebar to start.")
    st.stop()

# Load chain
chain = get_chain()

# ── Display chat history ──────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and "data" in msg:
            render_answer(msg["data"])
        else:
            st.markdown(msg["content"])

# ── Suggested starter questions ──────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("#### 💬 Try asking:")
    starters = [
        "What is the relationship between diabetes and renal failure?",
        "What are the symptoms of chronic kidney disease?",
        "How does obesity contribute to Type 2 diabetes?",
        "What is vanity sizing and how does it affect obesity perception?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(starters):
        if cols[i % 2].button(q, key=f"starter_{i}"):
            st.session_state.pending_question = q
            st.rerun()

# ── Handle pending question (from follow-up or starter buttons) ───────────────
if st.session_state.pending_question:
    user_q = st.session_state.pending_question
    st.session_state.pending_question = None
    st.session_state.messages.append({"role": "user", "content": user_q})

    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and generating structured answer…"):
            try:
                data = query(chain, user_q)
                render_answer(data)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "data": data,
                })
            except Exception as e:
                st.error(f"Error: {e}")
    st.rerun()

# ── Text input ────────────────────────────────────────────────────────────────
if user_input := st.chat_input("Ask a medical question…"):
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving and generating structured answer…"):
            try:
                data = query(chain, user_input)
                render_answer(data)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": data["answer"],
                    "data": data,
                })
            except Exception as e:
                st.error(f"Error: {e}\n\nMake sure you ran `python ingest.py` first.")