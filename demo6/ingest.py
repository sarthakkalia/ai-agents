import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

DOCS_DIR   = Path("./docs")
CHROMA_DIR = "./chroma_db" 
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_docs() -> list:
    all_docs = []
    pdfs = list(DOCS_DIR.glob("*.pdf"))

    if not pdfs:
        raise FileNotFoundError(
            f"No PDFs in '{DOCS_DIR}'. Add PDFs then rerun."
        )
    
    for pdf_path in pdfs:
        print(f"  Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs   = loader.load()

        for doc in docs:
            doc.metadata["file_name"] = pdf_path.name

        all_docs.extend(docs)
        print(f"  └─ {len(docs)} pages")

    print(f"\n[STEP 1] Total pages: {len(all_docs)}")
    return all_docs

def split(docs: list) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)
    print(f"[STEP 2] {len(chunks)} chunks  (size=600, overlap=100)")
    return chunks

def embed_and_store(chunks: list):
    print("[STEP 3] Loading embedding model (first run ~90MB download)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[STEP 3] Embedding {len(chunks)} chunks into ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"[STEP 3] Saved to '{CHROMA_DIR}/' ✓")
    return vectorstore

if __name__ == "__main__":
    print("=" * 50)
    print("  RAG Ingestion Pipeline")
    print("=" * 50)
    DOCS_DIR.mkdir(exist_ok=True)
    docs   = load_docs()
    chunks = split(docs)
    embed_and_store(chunks)
    print("\nDone! Now run:  python main.py")