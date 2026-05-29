import os
from pathlib import Path
from typing import Literal
from dotenv import load_dotenv
 
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
 
from embeddings import get_embeddings
from state import AgenticRAGState

load_dotenv()

CHROMA_DIR  = "./chroma_db"
MODEL_DIR   = "./models"

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)

def _load_retriever():
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError("Run python ingest.py first.")
    
    embeddings = get_embeddings()

    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    ).as_retriever(search_kwargs={"k": 4})
 
retriever     = _load_retriever()
tavily_search = TavilySearch(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY"),
    content_chars_max=2000,
)

# node-1 (retriever node)
def retrieve_node(state: AgenticRAGState) -> dict:
    print(f"\n  [retrieve] searching for: '{state['question']}'")

    docs = retriever.invoke(state["question"])
    print(f"  [retrieve] got {len(docs)} docs")

    original = state.get("original_question") or state["question"]

    return {"documents": docs, "original_question": original}

# node-2: (grade_docs_node) 
class GradeDoc(BaseModel):
    """Grade whether a document is relevant to the question."""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' if not"
    )

grader_llm = llm.with_structured_output(GradeDoc)
 
grader_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a strict relevance grader.\n"
     "Your job is to check if a document would ACTUALLY HELP answer the question.\n\n"
     "Grade 'yes' ONLY IF:\n"
     "  - The document is genuinely about the same topic as the question\n"
     "  - It contains information that directly addresses what is being asked\n\n"
     "Grade 'no' IF:\n"
     "  - The document only shares some keywords but is about a different subject\n"
     "  - The document is tangentially related but would not help answer the question\n"
     "  - The overlap is coincidental (e.g. 'generation' in animation ≠ 'generation' in RAG)\n\n"
     "Be strict. A false positive (grading irrelevant doc as relevant) is worse than a false negative."),
    ("human",
     "Question: {question}\n\n"
     "Document content:\n{document}\n\n"
     "Would this document actually help answer the question? Answer yes or no only."),
])

grader_chain = grader_prompt | grader_llm

def grade_docs_node(state: AgenticRAGState) -> dict:
    print(f"\n  [grade_docs] grading {len(state['documents'])} documents...")
    relevant = []
    for i, doc in enumerate(state["documents"]):
        result: GradeDoc = grader_chain.invoke({
            "document": doc.page_content,
            "question": state["question"],
        })
        source = doc.metadata.get("file_name", "?")
        print(f"  [grade_docs] doc {i+1} ({source}): {result.score}")
        if result.score == "yes":
            relevant.append(doc)
 
    print(f"  [grade_docs] {len(relevant)}/{len(state['documents'])} docs are relevant")
    return {"relevant_docs": relevant}

# node-3: (generate_node)
generate_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a helpful assistant. Answer the question using ONLY the provided context.\n"
     "If the context doesn't contain enough information, say so honestly.\n"
     "Do not make up information. Cite which source(s) you used.\n\n"
     "Context:\n{context}"),
    ("human", "{question}"),
])
generate_chain = generate_prompt | llm

def generate_node(state):
    docs_to_use = (
        state.get("relevant_docs")
        or state.get("web_search_context")
        or state.get("documents", [])
    )
 
    source_label = (
        "relevant ChromaDB docs" if state.get("relevant_docs")
        else "web search results" if state.get("web_search_context")
        else "raw ChromaDB docs"
    )
    attempts = state.get("generation_attempts", 0) + 1
    print(f"\n  [generate] attempt {attempts} — using {len(docs_to_use)} {source_label}")
 
    context = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('file_name', 'web')}]\n{d.page_content}"
        for d in docs_to_use
    ])
 
    response = generate_chain.invoke({
        "context":  context,
        "question": state.get("original_question") or state["question"],
    })
 
    return {
        "generation":           response.content,
        "generation_attempts":  attempts,
    }

# node-4: (rewrite_query_node)
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriter improving retrieval from a vector database.\n\n"
     "STRICT RULES:\n"
     "1. NEVER change the subject or domain of the original question\n"
     "2. If the original question is about machine learning, keep it in ML\n"
     "3. If the original question is about biology, keep it in biology\n"
     "4. Use web context ONLY for vocabulary — not to change the topic\n"
     "5. If web results are about a completely different topic, ignore them\n"
     "6. Keep the rewritten query under 12 words\n"
     "7. Return ONLY the rewritten question — no explanation\n\n"
     "EXAMPLE (what NOT to do):\n"
     "  Original: 'what is transformer' (ML context)\n"
     "  Web shows: electrical transformers\n"
     "  WRONG rewrite: 'what is electrical transformer device'   ← changed domain!\n"
     "  RIGHT rewrite: 'transformer neural network architecture attention mechanism'"),
    ("human",
     "Original question (MUST stay in this domain): {original_question}\n\n"
     "Web context (use for vocabulary only, do not change subject):\n"
     "{web_context}\n\n"
     "Rewrite the query with different words but SAME topic and domain:"),
])
rewrite_chain = rewrite_prompt | llm

def rewrite_query_node(state: AgenticRAGState) -> dict:
    print(f"\n  [rewrite_query] attempt {state.get('rewrite_count', 0) + 1}")
    print(f"  [rewrite_query] original: '{state.get('original_question', state['question'])}'")
 
    # Summarise web context into a short excerpt for the prompt
    web_docs = state.get("web_search_context", [])
    if web_docs:
        web_context = "\n\n".join([
            f"[URL: {d.metadata.get('url', 'web')}]\n{d.page_content[:300]}"
            for d in web_docs[:3]
        ])
    else:
        web_context = "No web context available."
 
    result = rewrite_chain.invoke({
        "original_question": state.get("original_question", state["question"]),
        "web_context":       web_context,
    })
    new_question = result.content.strip()
 
    print(f"  [rewrite_query] rewritten: '{new_question}'")
 
    return {
        "question":      new_question,
        "rewrite_count": state.get("rewrite_count", 0) + 1,
        "documents":     [],
        "relevant_docs": [],
    }

# node-5: (web_search_node)
def web_search_for_context_node(state: AgenticRAGState) -> dict:
    search_query = state.get("original_question") or state["question"]
    print(f"\n  [web_search_for_context] fetching web context for: '{search_query}'")
 
    results = tavily_search.invoke({"query": search_query})
    raw     = results if isinstance(results, list) else results.get("results", [])
 
    web_docs = [
        Document(
            page_content=r.get("content", ""),
            metadata={"file_name": "web", "url": r.get("url", "")},
        )
        for r in raw
    ]
    print(f"  [web_search_for_context] got {len(web_docs)} results → stored in web_search_context")

    return {"web_search_context": web_docs}


# node-6: (answer_check_node)

class AnswerCheck(BaseModel):
    """Check whether the generated answer actually answers the question."""
    useful: Literal["yes", "no"] = Field(
        description=(
            "'yes' if the answer directly addresses the question with real information. "
            "'no' if the answer says it lacks information, couldn't find details, "
            "or only provides vague/indirect information."
        )
    )
 
answer_check_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an answer quality checker.\n"
     "Decide if the provided answer actually answers the question or not.\n\n"
     "Answer 'no' (not useful) if the answer:\n"
     "  - Says 'the context does not contain' or 'not enough information'\n"
     "  - Gives a vague, indirect answer without real content\n"
     "  - Answers a different question than what was asked\n\n"
     "Answer 'yes' (useful) if the answer:\n"
     "  - Directly addresses the question with real information\n"
     "  - Provides a clear explanation even if partial"),
    ("human",
     "Question: {question}\n\n"
     "Answer to evaluate:\n{generation}\n\n"
     "Is this a useful answer?"),
])

def check_answer_node(state: AgenticRAGState) -> dict:
    print(f"\n  [check_answer] evaluating answer quality...")
    generation = state.get("generation", "")
    failure_phrases = [
        "does not contain enough information",
        "not enough information to answer",
        "cannot provide an answer",
        "no information available",
        "context is insufficient",
        "unable to answer this question",
    ]

    first_part    = generation[:200].lower()
    is_short      = len(generation.strip()) < 120
    obvious_fail  = any(
        p in first_part or (is_short and p in generation.lower())
        for p in failure_phrases
    )
 
    if obvious_fail:
        print(f"  [check_answer] obvious failure → needs better source")
        return {"answer_useful": False}
 
    answer_check_llm   = llm.with_structured_output(AnswerCheck)
    answer_check_chain = answer_check_prompt | answer_check_llm
 
    result = answer_check_chain.invoke({
        "question":   state.get("original_question") or state["question"],
        "generation": generation,
    })
    useful = result.useful == "yes"
    print(f"  [check_answer] answer useful: {result.useful}")
    return {"answer_useful": useful}