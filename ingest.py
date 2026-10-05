"""
Run this ONCE to load your knowledge base into ChromaDB.
Usage: python ingest.py
"""
import os
import re
import logging
from dotenv import load_dotenv
load_dotenv()

from vector_store import vector_store
from alerts import send_alert

logger = logging.getLogger(__name__)

KB_PATH = "knowledge_base.txt"


def load_sections(path: str) -> list[dict]:
    """
    Split by === SECTION === headers.
    Each section becomes one chunk — ideal for FAQ-style content.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found at: {path}")

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


def ingest(sync_alert: bool = True):
    """
    Ingest knowledge base sections into ChromaDB.
    Sends an alert if ingestion fails.
    """
    try:
        sections = load_sections(KB_PATH)
        if not sections:
            raise ValueError(f"No valid sections found in {KB_PATH}")

        texts = [s["content"] for s in sections]
        metadatas = [s["metadata"] for s in sections]

        # Clear existing collection first (safe re-ingest)
        vector_store.reset_collection()
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        print(f"Ingested {len(texts)} sections into ChromaDB:")
        for s in sections:
            print(f"   - {s['metadata']['section']}")
        return {"status": "success", "count": len(texts)}
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"Ingestion failed: {e}")
        send_alert(
            key="kb_ingest_fail",
            subject="Knowledge Base Ingestion Failed",
            body=f"Failed to ingest knowledge base ({KB_PATH}) into ChromaDB:\n\nError: {e}",
            sync=sync_alert,
        )
        raise


if __name__ == "__main__":
    try:
        ingest(sync_alert=True)
    except Exception:
        pass