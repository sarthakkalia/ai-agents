from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgenticRAGState(TypedDict):
    qn: str
    org_qn: str
    docs: list[Document]
    rel_docs: list[Document]
    retrieval_confidence: str
    web_search_cntx: list[Document]
    gen: str
    rewrite_query_cnt: int
    web_search_used: bool
    ans_useful: bool
    gen_attempts:  int
    msg: Annotated[list[BaseMessage], add_messages]
    pipeline_log: list[str]