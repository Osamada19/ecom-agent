"""
Run this ONCE to load your knowledge base into ChromaDB.
Usage: python ingest.py
"""
import re
import logging
from vector_store import vector_store

logger = logging.getLogger(__name__)

KB_PATH = "knowledge_base.txt"


def load_sections(path: str) -> list[dict]:
    """
    Split by === SECTION === headers.
    Each section becomes one chunk — ideal for FAQ-style content.
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on === HEADER === lines
    raw_sections = re.split(r"\n(?===)", text)
    sections = []
 
    for section in raw_sections:
        section = section.strip()
        if not section:
            continue

        # Extract header as metadata
        header_match = re.match(r"^=== (.+?) ===", section)
        header = header_match.group(1) if header_match else "GENERAL"

        sections.append({
            "content": section,
            "metadata": {"section": header},
        })

    return sections


def ingest():
    try:
        sections = load_sections(KB_PATH)
        if not sections:
            logger.error("No sections found in knowledge_base.txt")
            return

        texts = [s["content"] for s in sections]
        metadatas = [s["metadata"] for s in sections]

        # Clear existing collection first (safe re-ingest)
        vector_store.reset_collection()
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        logger.info(f"✅ Ingested {len(texts)} sections into ChromaDB")
        for s in sections:
            logger.info(f"   - {s['metadata']['section']}")
    except Exception as e:
        logger.error(f"❌ Ingest failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    ingest()
