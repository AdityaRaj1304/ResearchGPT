import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
CHROMA_DB_DIR = (DATA_DIR / "vector_store") if (DATA_DIR / "vector_store").exists() else (BASE_DIR / "chroma_db")

COLLECTIONS = {
    "wholedoc": "arxiv_wholedoc",
    "fixed": "arxiv_fixed_512",
    "overlap": "arxiv_overlap_512_128",
    "semantic": "arxiv_semantic_512",
}

ALL_COLLECTION_NAMES = list(COLLECTIONS.values())
STRATEGIES = ["dense", "hybrid", "hybrid_rerank"]

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
