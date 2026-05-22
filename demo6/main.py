import json
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import graph, auto_graph
from state import AgentState


sessions: dict[str, dict] = {}


def new_session(label: str = "") -> dict:
    tid = str(uuid.uuid4())[:8]
    sessions[tid] = {
        "label":      label or f"Session {len(sessions) + 1}",
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }
    config = {"configurable": {"thread_id": tid}}
    print(f"\n  New session: {tid}  ({sessions[tid]['label']})")
    return config


def thread_id(config: dict) -> str:
    return config["configurable"]["thread_id"]


def handle_interrupt(config: dict) -> str:
    snapshot = graph.get_state(config)
    last_msg = snapshot.values["messages"][-1]

    if not (hasattr(last_msg, "tool_calls") and last_msg.tool_calls):
        return "no_tool"

    print(f"\n  {'⏸ INTERRUPT ':─<46}")
    for tc in last_msg.tool_calls:
        tool_name = tc["name"]
        args      = tc["args"]

        if tool_name == "rag_search":
            label = "RAG (searching your documents)"
        else:
            label = "Web search (Tavily live search)"

        print(f"  Tool : {label}")
        print(f"  Query: {args.get('query', args)}\n")

    while True:
        choice = input("  Approve? [y / n / e=edit query] : ").strip().lower()

        if choice == "y":
            print("  ✓ Approved\n")
            return "approved"

        elif choice == "n":
            graph.update_state(
                config,
                {"messages": [HumanMessage(
                    content="That tool call was rejected. Please answer from what you know."
                )]},
                as_node="agent",
            )
            return "rejected"

        elif choice == "e":
            new_q = input("  New query: ").strip()
            last_msg.tool_calls[0]["args"]["query"] = new_q
            graph.update_state(config, {"messages": [last_msg]}, as_node="agent")
            print(f"  ✎ Updated to: '{new_q}'\n")
            return "edited"


def chat(question: str, config: dict):
    print(f"\n{'═'*52}")
    print(f"  Thread : {thread_id(config)}")
    print(f"  Q      : {question}")
    print(f"{'═'*52}")

    graph.invoke(
        {
            "messages":    [HumanMessage(content=question)],
            "query":       question,
            "tools_used":  [],
            "rag_sources": [],
            "iteration":   0,
        },
        config=config,
    )

    # HIL loop
    while True:
        snapshot = graph.get_state(config)
        if not snapshot.next:
            break
        if "tools" in snapshot.next:
            handle_interrupt(config)
            graph.invoke(None, config=config)
        else:
            break

    # Extract final answer
    final_state = graph.get_state(config)
    messages    = final_state.values.get("messages", [])
    tools_used  = final_state.values.get("tools_used", [])

    final_answer = ""
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    print(f"\n{'─'*52}")
    print(f"  Answer:\n\n  {final_answer}\n")
    print(f"  Tools used : {tools_used or ['none (answered directly)']}")

    # Show which RAG sources were retrieved
    rag_msgs = [m for m in messages if isinstance(m, ToolMessage) and "Source" in str(m.content)]
    if rag_msgs:
        print("\n  RAG sources used:")
        for line in str(rag_msgs[0].content).split("\n"):
            if line.startswith("[Source"):
                print(f"    {line}")

    print(f"{'─'*52}\n")


def chat_auto(question: str, config: dict):
    print(f"\n  [AUTO] {question}")
    result = auto_graph.invoke(
        {
            "messages":    [HumanMessage(content=question)],
            "query":       question,
            "tools_used":  [],
            "rag_sources": [],
            "iteration":   0,
        },
        config=config,
    )
    last = result["messages"][-1]
    print(f"\n  Answer: {last.content}\n")


def show_history(config: dict):
    snap = graph.get_state(config)
    if not snap or not snap.values:
        print("  No history.")
        return
    print(f"\n  {'─'*48}")
    for msg in snap.values.get("messages", []):
        if isinstance(msg, HumanMessage):
            print(f"\n  [YOU]   {msg.content}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"\n  [AI]    {str(msg.content)[:300]}")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content)[:120]
            print(f"  [TOOL]  {msg.name}: {preview}...")
    print(f"\n  {'─'*48}")



def main():
    print("\n" + "═"*52)
    print("  RAG + Search Agent  (LangGraph)")
    print("  Tools: rag_search · tavily_search")
    print("═"*52)
    print("  Commands: new | resume <id> | history | auto <q> | quit\n")

    config = new_session("Default")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "new":
            label  = input("  Session name: ").strip()
            config = new_session(label)
        elif user_input.lower().startswith("resume "):
            tid = user_input.split(" ", 1)[1].strip()
            if tid in sessions:
                config = {"configurable": {"thread_id": tid}}
                print(f"  Resumed: {tid}")
            else:
                print(f"  Unknown session. Known: {list(sessions.keys())}")
        elif user_input.lower() == "history":
            show_history(config)
        elif user_input.lower() == "sessions":
            for tid, info in sessions.items():
                marker = " ← current" if thread_id(config) == tid else ""
                print(f"  {tid}  {info['label']}{marker}")
        elif user_input.lower().startswith("auto "):
            question    = user_input[5:].strip()
            auto_config = new_session("auto")
            chat_auto(question, auto_config)
        else:
            chat(user_input, config)


if __name__ == "__main__":
    main()