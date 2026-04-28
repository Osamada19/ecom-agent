from langdetect import detect, LangDetectException
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
import os
import re
from dotenv import load_dotenv
load_dotenv()

from state import SupportState
from tools import ALL_TOOLS

# ── LLM (single brain) ────────────────────────────────────────────────────────
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=os.getenv('GROQ_API_KEY'))
llm_with_tools = llm.bind_tools(ALL_TOOLS)

# ── Tool executor ─────────────────────────────────────────────────────────────
tool_node = ToolNode(ALL_TOOLS)

# ── Custom Darija Detection ───────────────────────────────────────────────────
DARIJA_PATTERNS = [
    r"\b(salam|slm|ahlan|marhaba|labas|bghit|bghiti|wakha|chof|chofi|3afak|afak|l3afak|safi|wakha|yallah|inshallah|mashi|mashi mushkil|wakha|sme7 liya|shukran|baraka|zwin|zwina|mezyan|khayb|3ziz|hbib|daba|daba daba|wakha|chhal|bsh7al|fin|fayn|kifach|kifesh|3lash|wech|wash|wesh|dkhel|khrej|jib|wakha|sme7li|3ziz 3liya|bghit nswl|bghit nchri|bghit n3ref|wakha|safi|yallah)\b",
    r"[379]",  # Darija uses numbers for Arabic letters
    r"\b(ana|nti|nta|hiya|houwa|hna|ntoma|homa|hadak|hadik|dak|dik|chi|wahed|juj|tlata|rb3a|khamsa|ssta|sb3a|tmenya|ts3oud|3achra)\b",
]

def is_darija(text: str) -> bool:
    """Check if text is Moroccan Darija (Latin script with numbers)."""
    text_lower = text.lower()
    for pattern in DARIJA_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

# ── Language detection map ────────────────────────────────────────────────────
LANG_MAP = {
    "ar": "arabic",
    "fr": "french",
    "en": "english",
    "ca": "french",
    "es": "french",
}

SYSTEM_PROMPT = """You are RELIA, the customer support agent for Nour Store — a Moroccan online fashion boutique based in Casablanca.

## IDENTITY — NON-NEGOTIABLE
- You are ALWAYS the agent (RELIA). You NEVER speak as or simulate the customer.
- You work for this store. You represent it professionally.

## HOW TO ANSWER
- ALWAYS call a tool before answering. Never answer from memory or assumptions.
- For policies, shipping, returns, payments, availability → call search_knowledge_base.
- For a specific order number → call lookup_order.
- If tools return nothing useful → say exactly: "I don't have that information. Please contact our support team at +212-6XX-XXXXXX."

## CONTEXT TRACKING
- If you previously asked the user for information (like an order ID), and their next message is short (1-3 words or just a number), assume they are answering your question.
- Example: You asked "What's your order ID?" and they reply "1001" → treat "1001" as the order ID.
- Example: You asked "Which city?" and they reply "Casablanca" → treat "Casablanca" as the city.
- Do NOT ask again if the short reply clearly answers your pending question.

## ESCALATION
- If the customer asks for a human, is angry, or the issue is complex → call escalate_to_human immediately.
- If the user provides a greeting (e.g., 'Salam', 'Hi', 'Hello'), always respond with a friendly greeting first. Only trigger the escalation tool if the customer/user asks for a human, is angry, or the issue is complex.

## INPUT HANDLING
- If the message is unclear or too vague → ask ONE short clarifying question.
- If the message is completely off-topic (not related to orders/products/store) → politely redirect.

## LANGUAGE — CRITICAL
- Reply ONLY in: {language}
- If the user writes in Darija (Moroccan Arabic in Latin script with numbers like 3=ع, 7=ح, 9=ق) → reply in DARIJA, not English or Standard Arabic.
- If the user writes in Standard Arabic (Arabic script) → reply in Standard Arabic.
- If the user writes in French → reply in French.
- If the user writes in English → reply in English.
- NEVER switch languages. Match the user's language exactly.

## DARIJA EXAMPLES (reply like this when user writes in Darija):
- User: "Salam, bghit n3ref wach 3ndkom had l3abaya?"
- You: "Salam! Iwa, 3ndna l'abaya. Wash bghiti tchriha?"
- User: "Fin wselat lcommande dyali?"
- You: "Lcommande dyalek wselat l..."
- User: "Shukran bzaf!"
- You: "L3afass, mashi mushkil!"

## TONE
- Friendly, concise, professional. No long paragraphs. Max 4 sentences unless listing steps.
- In Darija: casual and warm, like a helpful friend."""


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
    """Detect language with Darija support."""
    text = state["messages"][-1].content

    # Check Darija first (custom patterns)
    if is_darija(text):
        return {"language": "darija"}

    # Fall back to langdetect
    try:
        code = detect(text)
        language = LANG_MAP.get(code, "english")
    except LangDetectException:
        language = "english"

    return {"language": language}


# ── Node 2.5: Slot Filling ────────────────────────────────────────────────────
def slot_filler(state: SupportState) -> dict:
    """Fill pending slots from short user replies."""
    pending_intent = state.get("pending_intent")
    pending_slots = state.get("pending_slots", {})
    last_message = state["messages"][-1].content.strip()

    if not pending_intent or not pending_slots:
        return {}

    # Check if any slot is still empty and user provided a short answer
    for slot_name, slot_value in pending_slots.items():
        if slot_value is None:
            # Short reply = likely filling this slot
            if len(last_message.split()) <= 3 or last_message.isdigit():
                pending_slots[slot_name] = last_message
                return {
                    "pending_slots": pending_slots,
                    "messages": []  # Don't add new message, let router handle it
                }

    return {}


# ── Node 3: Router (the brain) ────────────────────────────────────────────────
def router_node(state: SupportState) -> dict:
    """Single LLM brain — decides to call tools or respond directly."""
    language = state.get("language") or "english"
    pending_intent = state.get("pending_intent")
    pending_slots = state.get("pending_slots", {})

    # Build context about pending slots
    slot_context = ""
    if pending_intent and pending_slots:
        empty_slots = [k for k, v in pending_slots.items() if v is None]
        filled_slots = {k: v for k, v in pending_slots.items() if v is not None}

        if filled_slots:
            slot_context = f"\n\nPENDING CONTEXT: You are handling \"{pending_intent}\". Already have: {filled_slots}."
        if empty_slots:
            slot_context += f" Still need: {empty_slots}."

    system = SYSTEM_PROMPT.format(language=language) + slot_context

    response = llm_with_tools.invoke(
        [SystemMessage(content=system)] + state["messages"]
    )

    # Track if we're asking for information (for next turn slot filling)
    new_pending = None
    new_slots = {}

    # Simple heuristic: if response asks for order ID, set pending intent
    content_lower = response.content.lower()
    if "order id" in content_lower or "order number" in content_lower or "numéro de commande" in content_lower:
        new_pending = "track_order"
        new_slots = {"order_id": None}
    elif "city" in content_lower or "ville" in content_lower:
        new_pending = "shipping_info"
        new_slots = {"city": None}

    updates = {"messages": [response]}
    if new_pending:
        updates["pending_intent"] = new_pending
        updates["pending_slots"] = new_slots

    return updates


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
            "سأقوم بتحويل طلبك إلى أحد الوكلاء البشريين على الفور. "
            "يرجى التواصل معنا عبر واتساب على +212-6XX-XXXXXX "
            "(من الاثنين إلى السبت، من 9 صباحًا حتى 6 مساءً بتوقيت المغرب). سنرد خلال ساعتين."
        ),
        "darija": (
            "غادي نحول ليك مع واحد من الفريق دابا. "
            "تواصل معانا على واتساب: +212-6XX-XXXXXX "
            "(من الاثنين للسبت، 9 الصباح حتى 6 المغرب). غادي يردو عليك خلال ساعتين."
        ),
    }

    return {
        "escalate": True,
        "pending_intent": None,
        "pending_slots": {},
        "messages": [AIMessage(content=messages.get(language, messages["english"]))],
    }