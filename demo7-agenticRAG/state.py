from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document

class AgenticRAGState(TypedDict):
    question: str
    original_question: str
    documents: list[Document]
    relevant_docs: list[Document]
    web_search_context: list[Document]
    generation: str
    rewrite_count: int
    web_search_used: bool
    answer_useful: bool
    generation_attempts:  int