import os
from dotenv import load_dotenv

# Load configuration from environment variables (.env)
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
LLM_MODEL = os.getenv("LLM_MODEL", "llama3")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_documents")
