from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    language: str          # "english" | "french" | "arabic"
    escalate: bool         # True when human takeover is needed
