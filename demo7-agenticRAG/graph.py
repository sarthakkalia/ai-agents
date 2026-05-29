from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import AgenticRAGState
from nodes import (
    retrieve_node,
    grade_docs_node,
    generate_node,
    rewrite_query_node,
    web_search_for_context_node,
    check_answer_node,
)

MAX_GENERATION_ATTEMPTS = 2
MAX_REWRITES = 2

def route_after_grading(state: AgenticRAGState) -> str:
    relevant_count   = len(state.get("relevant_docs", []))
    rewrite_count    = state.get("rewrite_count", 0)
    has_web_context  = bool(state.get("web_search_context"))
 
    if relevant_count > 0:
        print(f"\n  [router] {relevant_count} relevant docs → generate")
        return "generate"
 
    if not has_web_context:
        print(f"\n  [router] no relevant docs, no web context → web_search_for_context")
        return "web_search_for_context"
 
    if rewrite_count < MAX_REWRITES:
        print(f"\n  [router] no relevant docs, web cached, rewrites={rewrite_count} → rewrite_query")
        return "rewrite_query"
 
    print(f"\n  [router] max rewrites → generate from web context")
    return "generate"
    
def route_after_check(state: AgenticRAGState) -> str:
    generation_attempts = state.get("generation_attempts", 0)
    if generation_attempts >= MAX_GENERATION_ATTEMPTS:
        print(f"\n  [router] max generation attempts ({generation_attempts}) → END")
        return "end"
 
    # ── Normal routing ────────────────────────────────────────────────────────
    if state.get("answer_useful", True):
        print(f"\n  [router] answer is useful → END")
        return "end"
 
    has_web = bool(state.get("web_search_context"))
    if has_web:
        print(f"\n  [router] answer not useful, web context exists → regenerate from web")
        return "generate_from_web"
    else:
        print(f"\n  [router] answer not useful, no web context → fetch web then generate")
        return "web_search_for_context"

def build_graph() -> StateGraph[AgenticRAGState]:
    builder = StateGraph(AgenticRAGState)

    builder.add_node("retrieve",               retrieve_node)
    builder.add_node("grade_docs",             grade_docs_node)
    builder.add_node("generate",               generate_node)
    builder.add_node("check_answer",           check_answer_node)
    builder.add_node("web_search_for_context", web_search_for_context_node)
    builder.add_node("rewrite_query",          rewrite_query_node)
 
    def clear_relevant_and_generate(state):
        """Clears relevant_docs so generate_node uses web_search_context."""
        return {"relevant_docs": []}

    builder.add_node("clear_for_web", clear_relevant_and_generate)


    # fixed edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade_docs")
    builder.add_edge("generate", "check_answer")
    builder.add_edge("web_search_for_context", "rewrite_query")
    builder.add_edge("rewrite_query",          "retrieve")
    builder.add_edge("clear_for_web",          "generate")

    # conditional edges
    builder.add_conditional_edges(
        "grade_docs",
        route_after_grading,
        {
            "generate": "generate",
            "web_search_for_context": "web_search_for_context",
            "rewrite_query": "rewrite_query",
        },
    )
    
    builder.add_conditional_edges(
        "check_answer",
        route_after_check,
        {
            "end":                    END,
            "generate_from_web":      "clear_for_web",
            "web_search_for_context": "web_search_for_context",
        },
    )

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)

graph = build_graph()
