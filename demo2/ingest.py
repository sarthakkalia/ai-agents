"""
ingest.py  ─  One-time PDF ingestion script
=============================================
Run this ONCE to load your PDFs into Pinecone.
After this, app.py just queries Pinecone — no re-ingestion needed.

Usage:
    python ingest.py

What it does:
  1. Reads all PDFs from the  ./pdfs/  folder
  2. Splits them into overlapping chunks
  3. Embeds each chunk with HuggingFace (FREE, runs locally)
  4. Uploads chunk + vector to Pinecone

PINECONE SETUP (do this first — it's free):
  1. Go to https://app.pinecone.io  and create a free account
  2. Click "Create Index"
       Name:       medical-rag          (or whatever you want)
       Dimensions: 384                  ← MUST match all-MiniLM-L6-v2 output
       Metric:     cosine
       Cloud:      AWS  /  Region: us-east-1  (free tier)
  3. Copy your API key from the left sidebar
  4. Paste it into your .env file
"""

import os
import time
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, NLTKTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore


load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-rag")
PDF_FOLDER       = Path("./pdfs")

# all-MiniLM-L6-v2 produces 384-dimensional vectors
EMBEDDING_DIM    = 384
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdfs(folder: Path) -> list:
    """
    Loads every PDF in the folder.
    Returns a flat list of LangChain Document objects.
    Each Document has:
        .page_content  →  text of that page
        .metadata      →  {"source": "filename.pdf", "page": 0, ...}
    """
    all_docs = []
    pdf_files = list(folder.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs found in '{folder}'. "
            "Create a pdfs/ folder and drop your files there."
        )

    for pdf_path in pdf_files:
        print(f" Loading: {pdf_path.name}")
        loader = PyPDFLoader(str(pdf_path))
        docs   = loader.load()

        # Tag every chunk with which PDF it came from
        for doc in docs:
            doc.metadata["file_name"] = pdf_path.name

        all_docs.extend(docs)
        print(f"     → {len(docs)} pages loaded")

    print(f"\n[STEP 1] Total pages loaded: {len(all_docs)}")
    return all_docs

def split_documents(docs: list) -> list:
    """
    Splits full pages into smaller, overlapping chunks.

    chunk_size=500:
        Good balance for medical text — keeps a full paragraph together.
        Too small → loses context; too large → retrieves too much noise.

    chunk_overlap=100:
        Last 100 chars of chunk N appear at the start of chunk N+1.
        Prevents a key sentence from being cut off at a boundary.

    add_start_index=True:
        Stores the character position inside the original document.
        Useful for tracing exactly where a chunk came from.
    """
    NLTKsplitter = NLTKTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    RCTsplitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = NLTKsplitter.split_documents(docs)
    print(f"[STEP 2] Split into {len(chunks)} chunks")
    return chunks


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    Loads the sentence-transformer model locally.
    First run downloads ~90MB model weights; subsequent runs use cache.

    WHY all-MiniLM-L6-v2?
      - Fast: embeds ~14,000 sentences/sec on CPU
      - Small: 22M params, ~90MB
      - Good: great semantic similarity for Q&A tasks
      - FREE: runs on your machine, no API calls

    Output: 384-dimensional float vector per chunk of text.
    """
    print("[STEP 3] Loading HuggingFace embedding model (first run downloads ~90MB)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},        # use "cuda" if you have a GPU
        encode_kwargs={"normalize_embeddings": True},  # normalise → better cosine similarity
    )
    print(f"Model '{EMBEDDING_MODEL}' ready ✓")
    return embeddings


def get_or_create_pinecone_index(pc: Pinecone) -> None:
    """
    Creates the Pinecone index if it doesn't already exist.

    NEW v6 SDK NOTES:
      - Import: `from pinecone import Pinecone`
      - Init:   `pc = Pinecone(api_key=...)`
      - ServerlessSpec = free tier, no pods needed
    """
    existing = [idx.name for idx in pc.list_indexes()]

    if INDEX_NAME not in existing:
        print(f"[STEP 4] Creating Pinecone index '{INDEX_NAME}'...")
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIM,       # MUST match your embedding model output
            metric="cosine",               # cosine similarity works best for text
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1",        # free tier region
            ),
        )
        # Wait for the index to be ready
        print("Waiting for index to be ready", end="")
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            print(".", end="", flush=True)
            time.sleep(1)
        print(" ✓")
    else:
        print(f"[STEP 4] Index '{INDEX_NAME}' already exists, skipping creation.")


def upload_to_pinecone(chunks: list, embeddings: HuggingFaceEmbeddings, pc: Pinecone) -> PineconeVectorStore:
    """
    Embeds every chunk and uploads (vector, text, metadata) to Pinecone.

    PineconeVectorStore.from_documents():
      - Takes chunks + embedding model
      - Calls embeddings.embed_documents(texts) in batches
      - Upserts each (id, vector, metadata) into your Pinecone index
    """
    print(f"[STEP 5] Embedding {len(chunks)} chunks and uploading to Pinecone...")
    print(" (this may take 1–3 minutes depending on your internet speed)")

    index = pc.Index(INDEX_NAME)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME,
        pinecone_api_key=PINECONE_API_KEY,
    )
    print(f"Upload complete ✓  —  {len(chunks)} vectors in Pinecone")
    return vector_store


if __name__ == "__main__":
    print("=" * 55)
    print("  Medical RAG — Ingestion Pipeline")
    print("=" * 55)

    # Validate env vars before doing any work
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY not set. Check your .env file.")

    # Create pdfs/ folder if it doesn't exist
    PDF_FOLDER.mkdir(exist_ok=True)

    # Run the pipeline
    docs         = load_pdfs(PDF_FOLDER)
    chunks       = split_documents(docs)
    embeddings   = get_embeddings()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    get_or_create_pinecone_index(pc)

    vector_store = upload_to_pinecone(chunks, embeddings, pc)

    print("\n Ingestion complete!")
    print(f"   Index '{INDEX_NAME}' is ready to query.")
    print("   Run:  streamlit run app.py")