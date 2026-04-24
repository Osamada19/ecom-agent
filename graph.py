import sqlite3
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from state import SupportState
from nodes import (
    input_guard,
    detect_language,
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


# ── Routing: after tool execution ─────────────────────────────────────────────
def route_after_tools(state: SupportState) -> str:
    """Check if escalation tool was triggered."""
    for msg in reversed(state["messages"]):
        content = getattr(msg, "content", "")
        if isinstance(content, str) and "__ESCALATE__" in content:
            return "escalate"
    return "router"


# ── Build graph ───────────────────────────────────────────────────────────────
def build_graph():
    builder = StateGraph(SupportState)

    # Nodes
    builder.add_node("input_guard",     input_guard)
    builder.add_node("detect_language", detect_language)
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

    # Language → router
    builder.add_edge("detect_language", "router")

    # Router → tools or END
    builder.add_conditional_edges(
        "router",
        tools_condition,           # "tools" if tool_calls exist, "end" otherwise
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

    # Persistent memory
    conn = sqlite3.connect("memory.db", check_same_thread=False)
    memory = SqliteSaver(conn)

    return builder.compile(checkpointer=memory)


graph = build_graph()
