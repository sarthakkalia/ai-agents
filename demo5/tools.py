from langchain_community.tools import WikipediaQueryRun, tool
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_tavily import TavilySearch
import os
from dotenv import load_dotenv

load_dotenv()

wiki = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=2000,
    )
)

@tool
def safe_wikipedia(query: str):
    """Search Wikipedia safely."""
    try:
        return wiki.run(query)
    except Exception as e:
        return f"Wikipedia search failed: {str(e)}"

tavily_tool = TavilySearch(
    max_results=2,
    api_key=os.getenv("TAVILY_API_KEY"),
    content_chars_max=2000,
)

tools = [safe_wikipedia, tavily_tool]
