import os
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq

from state import AgentState
from rag_tool import rag_search
from search_tool import tavily_tool
 
load_dotenv()

tools = [rag_search, tavily_tool]

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0,
)
llm_with_tools = llm.bind_tools(tools)

def agent_node(state: AgentState) -> dict:
    print(f"\n  [agent] iteration {state.get('iteration', 0) + 1}")

    system = {
        "role": "system",
        "content": (
            "You are a research assistant with two tools:\n\n"
 
            "1. rag_search — searches uploaded internal documents.\n"
            "   Use this when the question is about topics likely covered\n"
            "   in the user's documents (medical info, reports, research papers).\n\n"
 
            "2. tavily_search_results_json — searches the live web.\n"
            "   Use this for current events, news, real-time data, or anything\n"
            "   not likely in the uploaded documents.\n\n"
 
            "When using rag_search, always cite the source document and page.\n"
            "When using web search, cite the URL.\n"
            "If you used rag_search but the results were poor, try web search."
        ),
    }
    response = llm_with_tools.invoke([system] + state["messages"])

    new_tools = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        new_tools = [tc["name"] for tc in response.tool_calls]
        print(f"  [agent] tool selected: {new_tools}")
 
    return {
        "messages":   [response],
        "iteration":  state.get("iteration", 0) + 1,
        "tools_used": state.get("tools_used", []) + new_tools,
    }


tools_node = ToolNode(tools)

def build_graph(interrupt: bool = True):
    builder = StateGraph(AgentState)
 
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
 
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
 
    memory = MemorySaver()
 
    kwargs = {"checkpointer": memory}
    if interrupt:
        kwargs["interrupt_before"] = ["tools"]
 
    return builder.compile(**kwargs)
 
 
graph      = build_graph(interrupt=True)
auto_graph = build_graph(interrupt=False)
 