import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
import traceback
from graph import graph

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RELIA — Store Support",
    page_icon="🛍️",
    layout="centered",
)

st.title("🛍️ RELIA — Customer Support")
st.caption("Ask about shipping, returns, orders, or anything about our store.")

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "escalated" not in st.session_state:
    st.session_state.escalated = False

# ── Display history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── Input ─────────────────────────────────────────────────────────────────────
if st.session_state.escalated:
    st.warning(
        "Your case has been escalated to a human agent. "
        "Please contact us on WhatsApp: **+212-6XX-XXXXXX**"
    )
else:
    user_input = st.chat_input("Type your message...")

    if user_input:
        # Show user message
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        # Run agent
        config = {"configurable": {"thread_id": st.session_state.thread_id}}

        with st.chat_message("assistant"):
            with st.spinner("RELIA is thinking..."):
                try:
                    result = graph.invoke(
                        {"messages": [HumanMessage(content=user_input)]},
                        config=config,
                    )

                    # Get last AI message
                    ai_messages = [
                        m for m in result["messages"]
                        if isinstance(m, AIMessage)
                    ]
                    response = ai_messages[-1] if ai_messages else AIMessage(
                        content="Something went wrong. Please try again."
                    )

                    st.markdown(response.content)
                    st.session_state.messages.append(response)

                    # Check escalation
                    if result.get("escalate"):
                        st.session_state.escalated = True
                        st.rerun()

                except Exception as e:
                    import traceback
                    error_text = traceback.format_exc()
                    with open("error.log", "w") as f:
                        f.write(error_text)
                    st.markdown("error logged")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Session")
    st.code(st.session_state.thread_id[:8] + "...", language=None)

    if st.button("🔄 New Conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.escalated = False
        st.rerun()

    st.markdown("---")
    st.markdown("**Support Hours**")
    st.markdown("Mon–Sat, 9am–6pm 🇲🇦")
    st.markdown("WhatsApp: +212-6XX-XXXXXX")
