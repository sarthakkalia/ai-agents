import os
from pathlib import Path
import re
from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_tavily import TavilySearch
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from sentence_transformers import CrossEncoder 
from embeddings import get_embeddings
from state import AgenticRAGState


load_dotenv()

CHROMA_DIR  = "./chroma_db"
MODEL_DIR   = "./models"

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="openai/gpt-oss-20b",
    temperature=0,
)


RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K  = 4
RETRIEVE_K    = 8
print("[rerank] loading cross-encoder model...")
cross_encoder = CrossEncoder(RERANK_MODEL)
print("[rerank] cross-encoder ready")

def _load_retriever():
    if not Path(CHROMA_DIR).exists():
        raise RuntimeError("Run python ingest.py first.")
    
    embeddings = get_embeddings()
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    ).as_retriever(search_kwargs={"k": RETRIEVE_K})
 
retriever     = _load_retriever()
tavily_search = TavilySearch(
    max_results=3,
    api_key=os.getenv("TAVILY_API_KEY"),
    content_chars_max=2000,
)

# reranker node
TOP_SCORE_THRESHOLD = -3.0
def rerank_node(state) -> dict:
    docs     = state["docs"]
    question = state["qn"]

    if not docs:
        print("  [rerank] no docs to rerank")
        return {"docs": []}
    
    print(f"\n  [rerank] scoring {len(docs)} docs with cross-encoder...")
    pairs  = [(question, doc.page_content) for doc in docs]

    scores = cross_encoder.predict(pairs)
    scored = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

    best_score = float(scored[0][0])
    print(f"  [rerank] best score={best_score:.3f}")

    if best_score < TOP_SCORE_THRESHOLD:
        print("  [rerank] low confidence retrieval")
        return {
            "docs": [],
            "retrieval_confidence": "low"
        }

    top_docs = [doc for _, doc in scored[:RERANK_TOP_K]]

    print(f"  [rerank] top {RERANK_TOP_K} scores after reranking:")
    for i, (score, doc) in enumerate(scored[:RERANK_TOP_K]):
        source = doc.metadata.get("file_name", "?")[:35]
        print(f"    {i+1}. score={score:.3f}  ({source})")
 
    discarded = len(docs) - RERANK_TOP_K
    if discarded > 0:
        print(f"  [rerank] discarded {discarded} low-scoring docs")

    return {"docs": top_docs, "retrieval_confidence": "high"}

# retriever node
def retrieve_node(state: AgenticRAGState) -> dict:
    print(f"\n  [retrieve] searching for: '{state['qn']}'")

    docs = retriever.invoke(state["qn"])
    print(f"  [retrieve] got {len(docs)} docs")

    original = state.get("org_qn") or state["qn"]

    return {"docs": docs, "org_qn": original}

# grade documents node
class GradeDoc(BaseModel):
    """Grade whether a document is relevant to the question."""
    score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant to the question, 'no' if not"
    )

grader_llm = llm.with_structured_output(GradeDoc,  method="json_mode")
 
grader_prompt = ChatPromptTemplate.from_messages([
(
    "system",
    """
    You are a relevance grader.
    Return JSON only.
    Valid responses:
    {{"score":"yes"}}
    or
    {{"score":"no"}}
    A document is relevant only if it helps answer the question.
    """
),
(
    "human",
    """
    Question:
    {question}
    Document:
    {document}
    """
)
])

grader_chain = grader_prompt | grader_llm

def grade_docs_node(state: AgenticRAGState) -> dict:
    print(f"\n  [grade_docs] grading {len(state['docs'])} documents...")
    relevant = []
    for i, doc in enumerate(state["docs"]):
        result: GradeDoc = grader_chain.invoke({
            "document": doc.page_content,
            "question": state["qn"],
        })
        source = doc.metadata.get("file_name", "?")
        print(f"  [grade_docs] doc {i+1} ({source}): {result.score}")
        if result.score == "yes":
            relevant.append(doc)
 
    print(f"  [grade_docs] {len(relevant)}/{len(state['docs'])} docs are relevant")
    return {"rel_docs": relevant}

# generate node
generate_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """
    You are a helpful research assistant.
    Previous conversation:
    {history}
    IMPORTANT RULES:
    1. Research papers rarely give textbook definitions.
    2. Synthesize information from the provided context.
    3. If multiple chunks describe a concept, combine them.
    4. Do NOT say "the context doesn't define it" if you can infer it.
    5. Only admit lack of information if the topic is genuinely absent.
    6. Cite sources at the end.

    Context:
    {context}
    """
        ),
        ("human", "{question}")
    ])

generate_chain = generate_prompt | llm

def generate_node(state):
    docs_to_use = (
        state.get("rel_docs")
        or state.get("web_search_cntx")
        or state.get("docs", [])
    )
 
    source_label = (
        "relevant ChromaDB docs"
        if state.get("rel_docs")
        else "web search results"
        if state.get("web_search_cntx")
        else "raw ChromaDB docs"
    )

    question = state.get("org_qn") or state["qn"]
    attempts = state.get("gen_attempts", 0) + 1
    print(f"\n  [generate] attempt {attempts} — using {len(docs_to_use)} {source_label}")
 
    print(
        f"\n  [generate] attempt {attempts} "
        f"— using {len(docs_to_use)} {source_label}"
    )

    context = "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('file_name', 'web')}]\n{d.page_content}"
        for d in docs_to_use
    ])

    prior_history = ""
    prior_messages = state.get("msg", [])
    if prior_messages:
        pairs = []
        for i in range(0, len(prior_messages) - 1, 2):
            if i + 1 < len(prior_messages):
                q = prior_messages[i].content
                a = prior_messages[i + 1].content[:300]
                pairs.append(f"Q: {q}\nA: {a}")
        if pairs:
            prior_history = "Previous conversation:\n" + "\n\n".join(pairs[-3:])

    response = generate_chain.invoke({
        "history": prior_history,
        "context": context,
        "question": question,
    })

    content = re.sub(r"<tool_call>.*?<tool_call>", "", response.content, flags=re.DOTALL).strip()
 
    new_messages = state.get("msg", []).copy()

    new_messages.extend([
        HumanMessage(content=question),
        AIMessage(content=content),
    ])

    log = state.get("pipeline_log", [])
    log.append(f"generate: attempt {attempts}, {len(docs_to_use)} docs used")


    return {
        "gen": content,
        "gen_attempts": attempts,
        "msg": new_messages,
        "pipeline_log": log,
        "web_search_cntx": [],
    }

# rewrite query node
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a query rewriter improving retrieval from a vector database.\n\n"
     "STRICT RULES:\n"
     "1. NEVER change the subject or domain of the original question\n"
     "2. If the original question is about machine learning, keep it in ML\n"
     "3. If the original question is about biology, keep it in biology\n"
     "4. Keep the rewritten query under 12 words\n"
     "5. Return ONLY the rewritten question — no explanation\n\n"),
    ("human",
    "Original question (MUST stay in this domain): {original_question}\n\n"
    "Rewrite the query SAME topic and domain:")
])
rewrite_chain = rewrite_prompt | llm

def rewrite_query_node(state: AgenticRAGState) -> dict:
    print(f"\n  [rewrite_query] attempt {state.get('rewrite_query_cnt', 0) + 1}")
    print(f"  [rewrite_query] original: '{state.get('org_qn', state['qn'])}'")
 
    result = rewrite_chain.invoke({
        "original_question": state.get("org_qn", state["qn"]),
    })
    new_question = result.content.strip()
 
    print(f"  [rewrite_query] rewritten: '{new_question}'")
 
    return {
        "qn":      new_question,
        "rewrite_query_cnt": state.get("rewrite_query_cnt", 0) + 1,
        "docs":     [],
        "rel_docs": [],
    }

# web search node
def web_search_for_context_node(state: AgenticRAGState) -> dict:
    search_query = state.get("org_qn") or state["qn"]
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
    print(f"  [web_search_for_context] got {len(web_docs)} results → stored in web search context")

    return {"web_search_cntx": web_docs, "web_search_used": True,}


# answer checking node
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
     "You are checking if an answer provides useful information about the topic.\n\n"
     "Answer 'yes' (useful) if the answer:\n"
     "  - Explains the concept even partially\n"
     "  - Describes how something works, even without a formal definition\n"
     "  - Synthesizes relevant information about the topic\n"
     "  - References relevant sources and gives context\n\n"
     "Answer 'no' (not useful) ONLY if the answer:\n"
     "  - Provides zero information about the actual topic\n"
     "  - Only says 'I cannot find this' with no explanation at all\n"
     "  - Answers a completely different question\n\n"
     "A partial or synthesized answer is USEFUL. "
     "Only completely empty answers are NOT useful."),
    ("human",
        """
        Question: {question}
        Answer:
        {generation}
        Return JSON only.
        Example:
        {{"useful":"yes"}}
        or
        {{"useful":"no"}}
        """
        ),
])

def check_answer_node(state: AgenticRAGState) -> dict:
    print(f"\n  [check_answer] evaluating answer quality...")
    generation = state.get("gen", "")
    failure_phrases = [
        "don't contain any information",
        "does not contain any information",
        "cannot determine",
        "cannot find information",
        "topic is not present",
    ]

    first_part    = generation[:200].lower()
    is_short      = len(generation.strip()) < 120
    obvious_fail  = any(
        p in first_part or (is_short and p in generation.lower())
        for p in failure_phrases
    )
 
    if obvious_fail:
        print(f"  [check_answer] obvious failure → needs better source")
        return {"ans_useful": False}
 
    answer_check_llm   = llm.with_structured_output(AnswerCheck, method="json_mode")
    answer_check_chain = answer_check_prompt | answer_check_llm
 
    result = answer_check_chain.invoke({
        "question":   state.get("org_qn") or state["qn"],
        "generation": generation,
    })
    useful = result.useful == "yes"
    print(f"  [check_answer] answer useful: {result.useful}")
    return {"ans_useful": useful}