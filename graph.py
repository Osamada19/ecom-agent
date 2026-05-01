import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from state import SupportState
from nodes import (
    input_guard,
    detect_language,
    slot_filler,
    router_node,
    tool_node,
    escalation_node,
)


# ── Routing: after guard ──────────────────────────────────────────────────────
def route_after_guard(state: SupportState) -> str:
    """If guard already replied (invalid input), go to END. Otherwise continue."""
    last = state["messages"][-1]
    from langchain_core.messages import AIMessage
    if isinstance(last, AIMessage):
        return "end"
    return "detect_language"


# ── Routing: after slot filler ────────────────────────────────────────────────
def route_after_slots(state: SupportState) -> str:
    """If slots were filled, go back to router. Otherwise continue to router normally."""
    pending_slots = state.get("pending_slots", {})
    # If all slots are filled, let router handle with complete info
    if pending_slots and all(v is not None for v in pending_slots.values()):
        return "router"
    return "router"  # Always go to router


# ── Routing: after tool execution ─────────────────────────────────────────────
def route_after_tools(state: SupportState) -> str:
    """Check if escalation tool was triggered in the LAST tool result only."""
    # Only check the most recent tool message, not all history
    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "tool":
            content = getattr(msg, "content", "")
            if isinstance(content, str) and "__ESCALATE__" in content:
                return "escalate"
            break  # Only check the last tool result
    return "router"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(SupportState)

    # Nodes
    builder.add_node("input_guard",     input_guard)
    builder.add_node("detect_language", detect_language)
    builder.add_node("slot_filler",     slot_filler)
    builder.add_node("router",          router_node)
    builder.add_node("tools",           tool_node)
    builder.add_node("escalation",      escalation_node)

    # Entry
    builder.add_edge(START, "input_guard")

    # After guard
    builder.add_conditional_edges(
        "input_guard",
        route_after_guard,
        {"end": END, "detect_language": "detect_language"},
    )

    # Language → slot filler → router
    builder.add_edge("detect_language", "slot_filler")
    builder.add_edge("slot_filler", "router")

    # Router → tools or END
    builder.add_conditional_edges(
        "router",
        tools_condition,
        {"tools": "tools", "__end__": END},
    )

    # After tools → check escalation or loop back to router
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {"escalate": "escalation", "router": "router"},
    )

    # Escalation always ends
    builder.add_edge("escalation", END)

    # Persistent memory with thread safety
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return builder.compile(checkpointer=memory)


graph = build_graph()