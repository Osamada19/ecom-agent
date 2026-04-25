from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEndpointEmbeddings
import os
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBEDDING_MODEL,
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

vector_store = Chroma(
    collection_name="store_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vector_store.as_retriever(
    search_type="mmr",            # MMR reduces redundant chunks
    search_kwargs={"k": 4, "fetch_k": 10},
)
