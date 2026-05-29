from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR   = "./models"


_MODEL_CACHE_PATH = Path(MODEL_DIR) / "models--sentence-transformers--all-MiniLM-L6-v2"

def get_embeddings() -> HuggingFaceEmbeddings:
    already_cached = _MODEL_CACHE_PATH.exists()

    if already_cached:
        print("  [embeddings] loading model from local cache")
    else:
        print("  [embeddings] downloading model for the first time...")

    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        cache_folder=MODEL_DIR,
        model_kwargs={
            "device": "cpu",
            "local_files_only": already_cached,
        },
        encode_kwargs={"normalize_embeddings": True},
    )

    