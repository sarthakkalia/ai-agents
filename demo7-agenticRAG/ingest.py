from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embeddings import get_embeddings
from langchain_chroma import Chroma
from pathlib import Path

DOCS_DIR    = Path("./docs")
CHROMA_DIR  = "./chroma_db"
# EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

MODEL_DIR   = "./models"

def ingest():
    pdfs = list(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDFs found in docs/. Add files and rerun.")
    
    all_docs = []
    for pdf in pdfs:
        print(f"  Loading: {pdf.name}")
        loader = PyPDFLoader(str(pdf))
        docs   = loader.load()
        for doc in docs:
            doc.metadata["file_name"] = pdf.name
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    chunks   = splitter.split_documents(all_docs)
    print(f"  {len(chunks)} chunks created")

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    print("Loading embedding model...")
    # embeddings = HuggingFaceEmbeddings(
    #     model_name=EMBED_MODEL,
    #     cache_folder=MODEL_DIR,
    #     model_kwargs={"device": "cpu"},
    #     encode_kwargs={"normalize_embeddings": True},
    # )

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    embeddings = get_embeddings()


    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    print(f"  Saved to {CHROMA_DIR}/ ✓\n  Now run: python main.py")

if __name__ == "__main__":
    DOCS_DIR.mkdir(exist_ok=True)
    ingest()