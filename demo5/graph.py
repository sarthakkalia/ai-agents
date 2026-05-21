import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from state import AgentState
from tools import tools

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="qwen/qwen3-32b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    print(f"\n  [agent_node] iteration {state.get('iteration', 0) + 1}")

    system_msg = {
        "role": "system",
        "content": (
            "You are a research assistant with two tools:\n"
            "1. wikipedia — use for factual background, definitions, history\n"
            "2. tavily_search — use for recent news, current events, live info\n\n"
            "Always cite which tool you used and why."
        )
    }

    all_messages = [system_msg] + state["messages"]
    response = llm_with_tools.invoke(all_messages)

    new_tools = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        new_tools = [tc["name"] for tc in response.tool_calls]

    return {
        "messages": [response],
        "iteration":  state.get("iteration", 0) + 1, 
        "tools_used": state.get("tools_used", []) + new_tools,
    }

tools_node = ToolNode(tools)

def build_graph(interrupt: bool = True):
    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        tools_condition,
    )

    builder.add_edge("tools", "agent")

    memory = MemorySaver()

    compile_kwargs = {
        "checkpointer": memory,
    }
    if interrupt:
        compile_kwargs["interrupt_before"] = ["tools"]
 
    graph = builder.compile(**compile_kwargs)
    return graph

graph = build_graph(interrupt=True)
auto_graph = build_graph(interrupt=False)

