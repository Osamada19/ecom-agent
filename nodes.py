from langdetect import detect, LangDetectException
from langchain_core.messages import SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
import os 
from dotenv import load_dotenv
load_dotenv()

from state import SupportState
from tools import ALL_TOOLS

# ── LLM (single brain) ────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv('GROQ_API_KEY'))
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ── Tool executor ─────────────────────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)

# ── Language detection map ────────────────────────────────────────────────────
LANG_MAP = {
    "ar": "arabic",
    "fr": "french",
    "en": "english",
    "ca": "french",   # langdetect sometimes misclassifies Darija as Catalan
    "es": "french",   # or Spanish — default to French for Moroccan context
}

SYSTEM_PROMPT = """You are RELIA, the customer support agent for a Moroccan e-commerce store.

## IDENTITY — NON-NEGOTIABLE
- You are ALWAYS the agent (RELIA). You NEVER speak as or simulate the customer.
- You work for this store. You represent it professionally.

## HOW TO ANSWER
- ALWAYS call a tool before answering. Never answer from memory or assumptions.
- For policies, shipping, returns, payments, availability → call search_knowledge_base.
- For a specific order number → call lookup_order.
- If tools return nothing useful → say exactly: "I don't have that information. Please contact our support team at +212-6XX-XXXXXX."

## ESCALATION
- If the customer asks for a human, is angry, or the issue is complex → call escalate_to_human immediately.

## INPUT HANDLING
- If the message is unclear or too vague → ask ONE short clarifying question.
- If the message is completely off-topic (not related to orders/products/store) → politely redirect.

## LANGUAGE
- Reply ONLY in: {language}
- If arabic → use Moroccan Darija (Arabic in Arabic script), not Modern Standard Arabic.
- Match the customer's tone: casual if they're casual, formal if formal.

## TONE
- Friendly, concise, professional. No long paragraphs. Max 4 sentences unless listing steps."""


# ── Node 1: Input Guard ───────────────────────────────────────────────────────
def input_guard(state: SupportState) -> dict:
    """Reject empty or non-text messages before hitting the LLM."""
    last = state["messages"][-1].content.strip()
    if len(last) < 2 or not any(c.isalpha() for c in last):
        return {
            "messages": [
                AIMessage(content="I didn't catch that. Could you please rephrase your question?")
            ]
        }
    return {}


# ── Node 2: Language Detection ────────────────────────────────────────────────
def detect_language(state: SupportState) -> dict:
    """Detect language without using an LLM."""
    text = state["messages"][-1].content
    try:
        code = detect(text)
        language = LANG_MAP.get(code, "english")
    except LangDetectException:
        language = "english"
    return {"language": language}


# ── Node 3: Router (the brain) ────────────────────────────────────────────────
def router_node(state: SupportState) -> dict:
    """Single LLM brain — decides to call tools or respond directly."""
    language = state.get("language") or "english"
    system = SYSTEM_PROMPT.format(language=language)

    response = llm_with_tools.invoke(
        [SystemMessage(content=system)] + state["messages"]
    )
    return {"messages": [response]}


# ── Node 4: Escalation Handler ────────────────────────────────────────────────
def escalation_node(state: SupportState) -> dict:
    """Triggered when escalate_to_human tool is called."""
    language = state.get("language") or "english"

    messages = {
        "english": (
            "I'm escalating your case to a human agent right away. "
            "Please reach us on WhatsApp at +212-6XX-XXXXXX "
            "(Mon–Sat, 9am–6pm Morocco time). We'll respond within 2 hours."
        ),
        "french": (
            "Je transmets votre demande à un agent humain immédiatement. "
            "Contactez-nous sur WhatsApp au +212-6XX-XXXXXX "
            "(Lun–Sam, 9h–18h heure du Maroc). Nous répondrons dans les 2 heures."
        ),
        "arabic": (
            "غادي نحول ليك مع واحد من الفريق دابا. "
            "تواصل معانا على واتساب: +212-6XX-XXXXXX "
            "(من الاثنين للسبت، 9 الصباح حتى 6 المغرب). غادي يردو عليك خلال ساعتين."
        ),
    }

    return {
        "escalate": True,
        "messages": [AIMessage(content=messages.get(language, messages["english"]))],
    }
