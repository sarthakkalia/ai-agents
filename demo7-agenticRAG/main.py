import uuid
from graph import graph
from state import AgenticRAGState

def ask(question: str, config: dict) -> dict:
    initial_state = {
        "qn":            question,
        "org_qn":   question,
        "docs":           [],
        "rel_docs":            [],
        "retrieval_confidence": "low",
        "web_search_cntx":     [],
        "gen":                 "",
        "rewrite_query_cnt":   0,
        "web_search_used":     False,
        "ans_useful":       False,
        "gen_attempts": 0,
        "msg":            [],
        "pipeline_log":        [],
    }

    print(f"\n{'═'*55}")
    print(f"  Question: {question}")
    print(f"{'═'*55}")

    final_state = graph.invoke(initial_state, config=config)

    print(f"\n{'─'*55}")
    print(f"  Answer:\n\n  {final_state['gen']}")
    print(f"\n  Rewrites needed : {final_state.get('rewrite_query_cnt', 0)}")
    print(f"  Web search used  : {final_state.get('web_search_used', False)}")
    print(f"  Relevant docs    : {len(final_state.get('rel_docs', []))}")
    print(f"{'─'*55}\n")


    log = final_state.get("pipeline_log", [])
    if log:
        print(f"\n  Pipeline log:")
        for entry in log:
            print(f"    → {entry}")

    print(f"{'─'*55}\n")
    return final_state


def main():
    print("\n" + "═"*55)
    print("  Agentic RAG")
    print("═"*55)
    print("  Type 'quit' to exit, 'history' to see conversation\n")

    import uuid
    config = {"configurable": {"thread_id": str(uuid.uuid4())[:8]}}
    print(f"  Session: {config['configurable']['thread_id']}\n")

    while True:
        try:
            q = input("Question > ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q.lower() == "quit":
            break
        if q.lower() == "history":
            snapshot = graph.get_state(config)
            msgs = snapshot.values.get("msg", [])
            print(f"\n  Conversation so far ({len(msgs)//2} turns):")
            for i in range(0, len(msgs), 2):
                if i + 1 < len(msgs):
                    print(f"\n  Q: {msgs[i].content}")
                    print(f"  A: {msgs[i+1].content[:200]}...")
            print()
            continue

        ask(q, config)


if __name__ == "__main__":
    main()