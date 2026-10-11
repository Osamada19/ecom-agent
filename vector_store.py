# Load environment variables (such as API keys) from the .env file so the app can access them.
from dotenv import load_dotenv
load_dotenv()

# Import required tools: operating system helpers and LangChain integrations for ChromaDB and Gemini.
import os
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Initialize the Google Gemini text embedding model.
# Embeddings convert words into lists of numbers (vectors) that capture their real meaning for semantic search.
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Connect to the local Chroma vector database where knowledge base chunks are stored.
# The data is saved to disk in the "./chroma_db" folder so it persists across server restarts.
vector_store = Chroma(
    collection_name="store_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

# Create a retriever to search the vector database using Maximum Marginal Relevance (MMR).
# MMR finds relevant information while filtering out repetitive results: it checks 10 candidates and returns the best 4 unique matches.
retriever = vector_store.as_retriever(
    search_type="mmr",
    # Fetch 10 relevant chunks first, then pick the 4 most diverse ones to avoid duplicate info
    search_kwargs={"k": 4, "fetch_k": 10},
)