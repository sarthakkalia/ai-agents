import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch

load_dotenv()

# Wikipedia tool
wikipedia_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=2000,
    )
)

# Tavily Search tool
tavily_search_tool = TavilySearch(
    api_key=os.getenv("TAVILY_API_KEY"),
    top_k_results=2,
)

tools = [wikipedia_tool, tavily_search_tool]
print(f"Tools loaded: {[t.name for t in tools]}")

llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",      
        temperature=0,
    )

#  graph agent
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt=(
        "You are a helpful research assistant. "
        "Use Wikipedia for factual background information. "
        "Use Tavily for recent news and current events. "
        "Always cite which tool you used."
    ),
)

print("Graph compiled ✓")
print()

def ask(question: str):
    print(f"Question: {question}")
    print("-" * 50)
    try:
        result = agent.invoke(
            {"messages": [("user", question)]}
        )
    except Exception as e:
        print(f"Error: {e}")
        return
    
    print("=" * 50)
    print()

    final_answer = result["messages"][-1].content
    print(f"Answer:\n{final_answer}")
    print()

    print("Tools used this run:")
    for msg in result["messages"]:
        if hasattr(msg, "name"):
            print(f"  → {msg.name}")
    print("=" * 50)
    print()

if __name__ == "__main__":
    ask("What is the capital of France?")
    ask("What are the latest developments in AI research?")
