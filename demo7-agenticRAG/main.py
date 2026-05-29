import uuid
from graph import graph
from state import AgenticRAGState


def ask(question: str) -> dict:
    """
    Runs the full agentic RAG pipeline for one question.
    Returns the final state so you can inspect everything.
    """
    config = {"configurable": {"thread_id": str(uuid.uuid4())[:8]}}

    initial_state: AgenticRAGState = {
        "question":       question,
        "documents":      [],
        "relevant_docs":  [],
        "generation":     "",
        "rewrite_count":  0,
        "web_search_used": False,
    }

    print(f"\n{'═'*55}")
    print(f"  Question: {question}")
    print(f"{'═'*55}")

    final_state = graph.invoke(initial_state, config=config)

    print(f"\n{'─'*55}")
    print(f"  Answer:\n\n  {final_state['generation']}")
    print(f"\n  Rewrites needed : {final_state.get('rewrite_count', 0)}")
    print(f"  Web search used  : {final_state.get('web_search_used', False)}")
    print(f"  Relevant docs    : {len(final_state.get('relevant_docs', []))}")
    print(f"{'─'*55}\n")

    return final_state


def main():
    print("\n" + "═"*55)
    print("  Agentic RAG  (grade → retry → fallback)")
    print("═"*55)
    print("  Type a question or 'quit'\n")

    while True:
        try:
            q = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q.lower() == "quit":
            break

        ask(q)


if __name__ == "__main__":
    main()