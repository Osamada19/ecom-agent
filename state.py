from typing import Annotated, Optional, TypedDict
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages


class SupportState(TypedDict):
    messages: Annotated[list, add_messages]
    language: str                    # "english" | "french" | "arabic" | "darija"
    escalate: bool                 # True when human takeover is needed
    pending_intent: Optional[str]  # e.g., "track_order", "return_item"
    pending_slots: dict            # e.g., {"order_id": None, "reason": None}
    last_tool_result: Optional[str]  # Track last tool output for context
    session_started: bool          # Track if this is a new session