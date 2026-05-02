from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

vector_store = Chroma(
    collection_name="store_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db",
)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 10},
)