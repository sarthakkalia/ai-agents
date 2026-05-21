from datetime import datetime
import json
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from graph import graph, auto_graph
from state import AgentState

sessions: dict[str, dict] = {}   # thread_id → {label, created_at}

def new_session(label: str = "") -> dict:
    """Creates a new thread_id and returns a LangGraph config dict."""
    thread_id = str(uuid.uuid4())[:8]   # short 8-char UUID for readability
    sessions[thread_id] = {
        "label":      label or f"Session {len(sessions) + 1}",
        "created_at": datetime.now().strftime("%H:%M:%S"),
    }
    config = {"configurable": {"thread_id": thread_id}}
    print(f"\n  New session: {thread_id}  ({sessions[thread_id]['label']})")
    return config


def get_config(thread_id: str) -> dict:
    """Returns a config dict for an existing thread_id."""
    return {"configurable": {"thread_id": thread_id}}


def get_thread_id(config: dict) -> str:
    return config["configurable"]["thread_id"]


def show_pending_tool_calls(config: dict) -> list:
    """
    After a HIL interrupt, reads the checkpoint to find what
    tool calls the agent requested but hasn't executed yet.

    graph.get_state(config) returns a StateSnapshot with:
      .values  → current AgentState dict
      .next    → tuple of node names that will run next
                 empty tuple () means graph is done
    """
    snapshot = graph.get_state(config)
    last_msg  = snapshot.values["messages"][-1]

    pending = []
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        for tc in last_msg.tool_calls:
            pending.append({
                "id":   tc["id"],
                "name": tc["name"],
                "args": tc["args"],
            })
    return pending


def show_history(config: dict):
    """Prints the full conversation history for a session."""
    snapshot = graph.get_state(config)
    if not snapshot or not snapshot.values:
        print("  No history yet.")
        return

    messages = snapshot.values.get("messages", [])
    print(f"\n  {'─'*50}")
    print(f"  Conversation history  ({len(messages)} messages)")
    print(f"  {'─'*50}")

    for msg in messages:
        if isinstance(msg, HumanMessage):
            print(f"\n  [YOU]  {msg.content}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"\n  [AI]   {msg.content[:300]}{'...' if len(str(msg.content)) > 300 else ''}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"  [AI → tool]  {tc['name']}({json.dumps(tc['args'])[:80]})")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content)[:150]
            print(f"  [TOOL] {msg.name}: {preview}...")

    meta = {
        "query":      snapshot.values.get("query", ""),
        "tools_used": snapshot.values.get("tools_used", []),
        "iteration":  snapshot.values.get("iteration", 0),
    }
    print(f"\n  Meta: {meta}")
    print(f"  {'─'*50}")



def handle_interrupt(config: dict) -> str:
    """
    Called when graph pauses before the tools node.
    Shows the pending tool call and asks for human decision.

    Returns: "approved" | "rejected" | "edited"
    """
    pending = show_pending_tool_calls(config)

    if not pending:
        return "no_tool"

    print(f"\n  {'⏸ INTERRUPT ':─<48}")
    print("  Agent wants to call a tool — review before executing:\n")

    for tc in pending:
        print(f"  Tool : {tc['name']}")
        print(f"  Args : {json.dumps(tc['args'], indent=4)}")
        print()

    while True:
        choice = input("  Approve? [y=yes / n=reject / e=edit args] : ").strip().lower()

        if choice == "y":
            print("  ✓ Approved — resuming...\n")
            return "approved"

        elif choice == "n":
            # Reject: add a user message telling the agent not to use that tool
            print("  ✗ Rejected — telling agent to answer without this tool.\n")
            graph.update_state(
                config,
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"Do NOT use the {pending[0]['name']} tool. "
                                "Please answer based on what you already know."
                            )
                        )
                    ]
                },
                as_node="agent",   # pretend this update came from the agent node
            )
            return "rejected"

        elif choice == "e":
            # Edit: let human change the query arg before the tool runs
            try:
                new_query = input(f"  New query (was: {pending[0]['args']}): ").strip()
                # update_state modifies the checkpoint directly
                # We update the last AIMessage's tool_calls with new args
                last_msg = graph.get_state(config).values["messages"][-1]
                updated_tool_calls = []
                for tc in last_msg.tool_calls:
                    updated_tool_calls.append({
                        **tc,
                        "args": {"query": new_query}
                    })
                last_msg.tool_calls = updated_tool_calls

                graph.update_state(
                    config,
                    {"messages": [last_msg]},
                    as_node="agent",
                )
                print(f"  ✎ Updated query to: '{new_query}' — resuming...\n")
                return "edited"
            except Exception as e:
                print(f"  Edit failed: {e}. Approving original.")
                return "approved"
        else:
            print("  Please enter y, n, or e.")


def chat(question: str, config: dict):
    """
    Runs one question through the agent with human-in-the-loop.

    THE FULL FLOW:
    1. graph.invoke() → runs agent_node → hits interrupt before tools
    2. We show the pending tool call to the human
    3. Human decides: approve / reject / edit
    4. graph.invoke(None, config) → resumes from checkpoint:
         - If approved: tools_node runs → agent_node loops → maybe more tools
         - If rejected: agent_node sees rejection message → answers without tool
    5. Loop until graph.get_state(config).next == () (graph is done)
    """
    thread_id = get_thread_id(config)
    print(f"\n{'═'*55}")
    print(f"  Thread: {thread_id}")
    print(f"  Q: {question}")
    print(f"{'═'*55}")

    # Initial input state
    initial_state = {
        "messages":   [HumanMessage(content=question)],
        "query":      question,
        "tools_used": [],
        "iteration":  0,
    }

    # ── First invoke: runs until interrupt (or END if no tool needed) ──────────
    # graph.invoke() with initial_state starts a fresh run on this thread
    # If the agent decides to call a tool → pauses at interrupt_before=["tools"]
    # If agent answers directly (no tool) → returns final state
    graph.invoke(initial_state, config=config)

    # ── HIL loop: keep going until graph is fully done ────────────────────────
    while True:
        snapshot = graph.get_state(config)

        # snapshot.next == () means graph has reached END
        if not snapshot.next:
            break

        # Graph is paused — there's a pending tool call
        if "tools" in snapshot.next:
            decision = handle_interrupt(config)

            # Resume from checkpoint.
            # Passing None as input means "no new input, just continue"
            # LangGraph picks up exactly where it left off
            graph.invoke(None, config=config)
        else:
            # Shouldn't happen, but break to avoid infinite loop
            break

    # ── Extract and print final answer ────────────────────────────────────────
    final_state = graph.get_state(config)
    final_msgs  = final_state.values.get("messages", [])

    # Walk backwards to find the last AIMessage with text content
    final_answer = ""
    for msg in reversed(final_msgs):
        if isinstance(msg, AIMessage) and msg.content:
            final_answer = msg.content
            break

    tools_used = final_state.values.get("tools_used", [])
    iterations = final_state.values.get("iteration", 0)

    print(f"\n{'─'*55}")
    print(f"  Answer:\n")
    print(f"  {final_answer}")
    print(f"\n  Tools used: {tools_used or ['none']}")
    print(f"  Iterations: {iterations}")
    print(f"{'─'*55}\n")


def chat_auto(question: str, config: dict):
    """Runs a question WITHOUT human-in-the-loop (fully autonomous)."""
    print(f"\n  [AUTO MODE] {question}")
    result = auto_graph.invoke(
        {
            "messages":   [HumanMessage(content=question)],
            "query":      question,
            "tools_used": [],
            "iteration":  0,
        },
        config=config,
    )
    last_msg = result["messages"][-1]
    print(f"\n  Answer: {last_msg.content}\n")



def main():
    print("\n" + "═"*55)
    print("  LangGraph Agent — Wikipedia + Tavily")
    print("  AgentState · Persistence · Human-in-the-loop")
    print("═"*55)
    print("  Commands: new | resume <id> | history | sessions | auto | quit")
    print()

    current_config = new_session("Default")

    while True:
        try:
            user_input = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        # ── Commands ──────────────────────────────────────────────────────────
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        elif user_input.lower() == "new":
            label = input("  Session name (optional): ").strip()
            current_config = new_session(label)

        elif user_input.lower().startswith("resume "):
            tid = user_input.split(" ", 1)[1].strip()
            if tid in sessions:
                current_config = get_config(tid)
                print(f"  Resumed: {tid} ({sessions[tid]['label']})")
            else:
                print(f"  Unknown thread_id: {tid}")
                print(f"  Known sessions: {list(sessions.keys())}")

        elif user_input.lower() == "history":
            show_history(current_config)

        elif user_input.lower() == "sessions":
            print("\n  Known sessions:")
            for tid, info in sessions.items():
                marker = " ← current" if get_thread_id(current_config) == tid else ""
                print(f"  {tid}  {info['label']}  (started {info['created_at']}){marker}")
            print()

        elif user_input.lower().startswith("auto "):
            question = user_input[5:].strip()
            auto_config = new_session("auto")
            chat_auto(question, auto_config)

        # ── Regular question (with HIL) ───────────────────────────────────────
        else:
            chat(user_input, current_config)


if __name__ == "__main__":
    main()