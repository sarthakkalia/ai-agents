from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    query: str
    tools_used: list[str]
    iteration: int