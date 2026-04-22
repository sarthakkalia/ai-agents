"""
rag_pipeline.py
================================================
This file handles the entire RAG pipeline:
  1. Load documents  (PDF → raw text)
  2. Chunk           (raw text → small pieces)
  3. Embed           (small pieces → vectors / numbers)
  4. Store           (vectors → ChromaDB)
  5. Retrieve        (user query → most relevant chunks)
  6. Generate        (chunks + query → final answer via LLM)
"""


import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# pine code
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()
GROQ_API_KEY = os.getenv("Groq_api_key")


def load_and_split_documents(file_path: str) -> list:
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Only .pdf and .txt files are supported right now.")
    documents = loader.load()
    print(f"[STEP 1] Loaded {len(documents)} page(s) from '{file_path}'")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"[STEP 2] Split into {len(chunks)} chunks  (size≈500, overlap=100)")
    return chunks


def create_vector_store(chunks: list, persist_dir: str = "chroma_store") -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("[STEP 3] Embedding model ready (all-MiniLM-L6-v2)")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
    )
    vector_store.persist()

    print(f"[STEP 4] Vector store saved to '{persist_dir}/'")
    return vector_store

def load_existing_vector_store(persist_dir: str = "chroma_store") -> Chroma:
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = Chroma(
        persist_directory=persist_dir,
        embedding=embedding_model,
    )

    print(f"[LOAD] Loaded existing vector store from '{persist_dir}/'")
    return vector_store

def build_rag_chain(vector_store: Chroma):
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4},
    )
    print("[STEP 5] Retriever ready  (top-4 chunks per query)")
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Use ONLY the context below to answer the question.
If the answer is not in the context, say "I don't have enough information to answer that."
Do NOT make up information.

Context:
---------
{context}
---------

Question: {question}

Helpful Answer:
""")
 
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="openai/gpt-oss-120b"
    )

    # Format documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("[STEP 6] RAG chain built (modern LCEL)")
    return rag_chain


def ask(chain, question: str):
    answer = chain.invoke(question)
    return answer, []