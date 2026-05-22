import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

CHROMA_DIR  = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _load_retriever():
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError(
            f"ChromaDB not found at '{CHROMA_DIR}'.\n"
            "Run:  python ingest.py   first."
        )
    
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )

    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": 4,
            "score_threshold": 0.3,
        },
    )

retriever = _load_retriever()

@tool
def rag_search(query: str) -> str:
    """
    Search internal documents for information.
 
    Use this tool when the question is about topics that might be
    covered in the uploaded documents (e.g. medical conditions,
    research papers, reports, or any domain-specific content).
 
    Do NOT use this for current events, news, or real-time information.
    """

    docs = retriever.invoke(query)

    if not docs:
        return (
            "No relevant information found in the documents. "
            "Try rephrasing or use the web search tool instead."
        )
    
    parts = []
    for i, doc in enumerate(docs, 1):
        file_name = doc.metadata.get("file_name", "unknown")
        page      = doc.metadata.get("page", "?")
        parts.append(
            f"[Source {i}: {file_name}, page {page}]\n"
            f"{doc.page_content}"
        )
    
    return "\n\n---\n\n".join(parts)