"""
Run this ONCE to load your knowledge base into ChromaDB.
Usage: python ingest.py
"""
# Import standard library tools for file paths, regex pattern matching, and error logging.
import os
import re
import logging
# Load environment variables from the .env file.
from dotenv import load_dotenv
load_dotenv()

# Import the vector database instance and alert dispatch utility from project files.
from vector_store import vector_store
from alerts import send_alert

# Create a logger instance to capture runtime information and exceptions.
logger = logging.getLogger(__name__)

# Relative file path to the store's knowledge base text document.
KB_PATH = "knowledge_base.txt"


# Reads the knowledge base text file and splits it into discrete sections based on header markers.
# Each section is converted into a dictionary chunk suitable for vector database storage.
def load_sections(path: str) -> list[dict]:
    """
    Split by === SECTION === headers.
    Each section becomes one chunk — ideal for FAQ-style content.
    """
    # Check if the target knowledge base file exists before attempting to open it.
    if not os.path.exists(path):
        raise FileNotFoundError(f"Knowledge base file not found at: {path}")

    # Read the file with UTF-8 encoding so international text and special characters load properly.
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split on === HEADER === lines
    # Use regex lookahead to split right before each '===' marker so headers remain with their content.
    raw_sections = re.split(r"\n(?===)", text)
    sections = []
 
    # Iterate over each raw text segment to clean it and extract its category header.
    for section in raw_sections:
        # Strip leading and trailing whitespace to keep chunks clean.
        section = section.strip()
        # Skip empty strings or whitespace-only sections.
        if not section:
            continue

        # Extract header as metadata
        # Find the section name enclosed between '=== ' and ' ==='.
        header_match = re.match(r"^=== (.+?) ===", section)
        # Fall back to 'GENERAL' if no header marker was found at the beginning.
        header = header_match.group(1) if header_match else "GENERAL"

        # Store the section text alongside its header metadata.
        sections.append({
            "content": section,
            "metadata": {"section": header},
        })

    return sections


# Main ingestion function that refreshes the vector database with up-to-date knowledge chunks.
# Resets the existing ChromaDB collection and indexes new text embeddings.
def ingest(sync_alert: bool = True):
    """
    Ingest knowledge base sections into ChromaDB.
    Sends an alert if ingestion fails.
    """
    try:
        # Load all structured sections from the knowledge base file.
        sections = load_sections(KB_PATH)
        # Ensure that at least one section was loaded successfully.
        if not sections:
            raise ValueError(f"No valid sections found in {KB_PATH}")

        # Separate the raw text bodies and metadata dictionaries into distinct lists.
        texts = [s["content"] for s in sections]
        metadatas = [s["metadata"] for s in sections]

        # Clear existing collection first (safe re-ingest)
        # Erase old records from the Chroma collection to prevent outdated or duplicate entries.
        vector_store.reset_collection()
        # Generate vector embeddings for each text and add them to the vector store.
        vector_store.add_texts(texts=texts, metadatas=metadatas)

        # Print a summary of successfully loaded sections to the console.
        print(f"Ingested {len(texts)} sections into ChromaDB:")
        for s in sections:
            print(f"   - {s['metadata']['section']}")
        return {"status": "success", "count": len(texts)}
    except Exception as e:
        # Log the error details with stack trace if ingestion encounters a problem.
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        print(f"Ingestion failed: {e}")
        # Send an immediate alert notification (e.g. to Telegram) reporting the failure.
        send_alert(
            key="kb_ingest_fail",
            subject="Knowledge Base Ingestion Failed",
            body=f"Failed to ingest knowledge base ({KB_PATH}) into ChromaDB:\n\nError: {e}",
            sync=sync_alert,
        )
        raise


# Allows running this script directly from the terminal to populate or refresh ChromaDB.
if __name__ == "__main__":
    try:
        # Run the ingestion workflow synchronously.
        ingest(sync_alert=True)
    except Exception:
        # Suppress exception re-raise on main exit since the alert and logger already handled it.
        pass