from langchain_tavily import TavilySearch
import os


tavily_tool = TavilySearch(
    max_results=2,
    api_key=os.getenv("TAVILY_API_KEY"),
    content_chars_max=2000,
)