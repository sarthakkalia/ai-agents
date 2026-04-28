import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from schemas import MedicalAnswer


load_dotenv()

GROQ_API_KEY     = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "medical-rag")
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"


def build_retriever(k: int = 5):
    print("[RAG] Loading embedding model for query...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"[RAG] Connecting to Pinecone index '{INDEX_NAME}'...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    print(f"[RAG] Retriever ready (top-{k} chunks per query) ✓")
    return retriever



def build_rag_chain(retriever):
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=1500,
    )

    structured_llm = llm.with_structured_output(MedicalAnswer)
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a medical information assistant specializing in diabetes,
renal failure, and obesity-related conditions.

Answer ONLY based on the provided context documents.
If the context doesn't contain enough information, say so honestly
and set confidence to "low".

You MUST respond with a structured JSON that matches the required schema exactly.
Do NOT add any text outside the JSON.

Context from medical documents:
================================
{context}
================================""",
        ),
        (
            "human",
            "Question: {question}"
        ),
    ])


    def format_docs(docs: list) -> str:
        parts = []
        for doc in docs:
            source = doc.metadata.get("file_name", "unknown")
            parts.append(f"Source: {source}\n{doc.page_content}")
        return "\n---\n".join(parts)

    chain = (
        {
            "context":  retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm
    )

    print("[RAG] LCEL chain built successfully.")
    return chain

def query(chain, question: str) -> dict:
    result: MedicalAnswer = chain.invoke(question)
    return result.to_display()