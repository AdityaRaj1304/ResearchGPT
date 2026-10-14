from typing import Dict, List, Literal, Optional, Any
from pydantic import BaseModel, Field

CollectionType = Literal[
    "arxiv_wholedoc",
    "arxiv_fixed_512",
    "arxiv_overlap_512_128",
    "arxiv_semantic_512",
]
StrategyType = Literal["dense", "hybrid", "hybrid_rerank"]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3, description="Academic research question")
    collection: CollectionType = Field(
        default="arxiv_semantic_512",
        description="Target ChromaDB chunking collection",
    )
    strategy: StrategyType = Field(
        default="hybrid_rerank",
        description="Retrieval strategy: dense, hybrid (BM25+RRF), or hybrid_rerank",
    )
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    generate_answer: bool = Field(
        default=True,
        description="Synthesize cited answer using Groq LLM",
    )


class RetrievedChunk(BaseModel):
    id: str
    document: str
    metadata: Dict[str, Any]
    score: float


class LatencyBreakdown(BaseModel):
    embedding_ms: float = 0.0
    dense_ms: float = 0.0
    sparse_bm25_ms: float = 0.0
    rrf_fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0


class QueryResponse(BaseModel):
    query: str
    collection: str
    strategy: str
    top_k: int
    answer: Optional[str] = None
    chunks: List[RetrievedChunk]
    timings: LatencyBreakdown


class CompareRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: int = Field(default=5, ge=1, le=10)
    collections: List[CollectionType] = Field(
        default=["arxiv_fixed_512", "arxiv_semantic_512"]
    )
    strategies: List[StrategyType] = Field(
        default=["dense", "hybrid", "hybrid_rerank"]
    )
