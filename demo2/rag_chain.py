"""
rag_chain.py  ─  The RAG pipeline (modern LCEL style)
======================================================
This is the core of the project.

OLD LangChain style (avoid):
    chain = RetrievalQA.from_chain_type(llm=..., retriever=...)

NEW LangChain LCEL style (use this):
    chain = retriever | format_docs | prompt | llm.with_structured_output(Schema)

LCEL = LangChain Expression Language
  - Uses  |  (pipe) operator — just like Unix pipes
  - Each step receives the output of the previous step
  - Fully composable, inspectable, and async-ready
  - Works with .invoke(), .stream(), .batch(), .ainvoke()

This file builds two things:
  1. build_retriever()  — connects to Pinecone and returns a retriever
  2. build_rag_chain()  — wires retriever → prompt → Groq → Pydantic output
"""

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
    """
    Connects to the Pinecone index created by ingest.py
    and returns a retriever that finds the top-k similar chunks.

    HOW RETRIEVAL WORKS:
      1. User question → embed with same HF model used in ingest.py
      2. Query Pinecone for top-k vectors closest to question vector
      3. Return those chunks as Document objects

    k=5: retrieve 5 chunks. More k → more context but slower + noisier.
         For medical Q&A, 4–6 is usually the sweet spot.
    """
    print("[RAG] Loading embedding model for query...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Connect to existing Pinecone index (must already be populated by ingest.py)
    print(f"[RAG] Connecting to Pinecone index '{INDEX_NAME}'...")
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=PINECONE_API_KEY,
    )

    # as_retriever() wraps the vector store in a Retriever interface
    # that LangChain chains expect
    retriever = vector_store.as_retriever(
        search_type="similarity",          # cosine similarity
        search_kwargs={"k": k},
    )
    print(f"[RAG] Retriever ready (top-{k} chunks per query) ✓")
    return retriever



def build_rag_chain(retriever):
    """
    Builds the full RAG chain using LCEL (pipe operator).

    THE CHAIN (read left to right):
        {question, context}
              ↓
           prompt              ← formats question + retrieved docs
              ↓
        llm.with_structured_output(MedicalAnswer)
                                ← Groq forced to return valid MedicalAnswer JSON
              ↓
           MedicalAnswer        ← Pydantic-validated Python object ✓

    with_structured_output():
      - Passes the Pydantic schema as a JSON Schema to the LLM
      - LLM generates JSON that matches the schema
      - LangChain automatically parses + validates it into a MedicalAnswer object
      - If validation fails → raises ValidationError (not a silent bad answer)
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model="llama-3.3-70b-versatile",   # best free Groq model for reasoning
        temperature=0,                      # 0 = deterministic, good for factual Q&A
        max_tokens=1500,
    )

    # --- Attach structured output ---
    # This is the KEY step: wraps the LLM so it MUST return a MedicalAnswer object.
    # Internally LangChain:
    #   1. Converts MedicalAnswer to JSON Schema
    #   2. Adds it to the API call as a "tool" or "response_format"
    #   3. Parses & validates the LLM response back into MedicalAnswer
    structured_llm = llm.with_structured_output(MedicalAnswer)

    # --- Prompt template ---
    # ChatPromptTemplate creates a [system, human] message pair.
    # {context} and {question} are placeholders filled at runtime.
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

    # --- Helper: format retrieved docs into a single string ---
    def format_docs(docs: list) -> str:
        """
        Converts a list of Document objects into one context string.
        Each chunk is separated by a divider and labelled with its source file.

        Input:  [Document(page_content="...", metadata={"file_name": "diabetes.pdf"}), ...]
        Output: "Source: diabetes.pdf\n...\n---\nSource: renal.pdf\n...\n---"
        """
        parts = []
        for doc in docs:
            source = doc.metadata.get("file_name", "unknown")
            parts.append(f"Source: {source}\n{doc.page_content}")
        return "\n---\n".join(parts)

    # ── LCEL chain assembly ───────────────────────────────────────────────────
    #
    # RunnablePassthrough() passes the input dict through unchanged.
    # RunnableLambda(fn) wraps any function as a chain step.
    #
    # Chain breakdown:
    #
    #  input: {"question": "What is HbA1c?"}
    #     ↓
    #  Step 1: retriever fetches top-5 docs for the question
    #  Step 2: format_docs joins them into one string → {context}
    #  Step 3: prompt.format({context, question}) → ChatPromptValue
    #  Step 4: structured_llm generates + validates MedicalAnswer
    #
    chain = (
        {
            # retriever.invoke(question) → list of Documents → formatted string
            "context":  retriever | RunnableLambda(format_docs),
            # pass the question through unchanged
            "question": RunnablePassthrough(),
        }
        | prompt
        | structured_llm     # returns a validated MedicalAnswer object
    )

    print("[RAG] LCEL chain built ✓")
    return chain


# ─────────────────────────────────────────────────────────────────────────────
# Query function — called by app.py
# ─────────────────────────────────────────────────────────────────────────────
def query(chain, question: str) -> dict:
    """
    Runs the full RAG chain for one question.

    Returns a display-ready dict via MedicalAnswer.to_display().
    All fields are guaranteed valid by Pydantic — no surprises.
    """
    # chain.invoke() runs all steps synchronously
    # Returns a MedicalAnswer Pydantic object (not a raw string!)
    result: MedicalAnswer = chain.invoke(question)

    # Convert to a plain dict for easy rendering in Streamlit
    return result.to_display()