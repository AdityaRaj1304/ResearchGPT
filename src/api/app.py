try:
    import torch
except Exception:
    pass

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any, Tuple, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

from src.config import (
    ALL_COLLECTION_NAMES,
    CHROMA_DB_DIR,
    GROQ_API_KEY,
    LLM_MODEL,
    PROCESSED_DIR,
)
from src.models.advanced_retriever import ParameterizedRetriever
from src.api.schemas import (
    CompareRequest,
    LatencyBreakdown,
    QueryRequest,
    QueryResponse,
    RetrievedChunk,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("researchgpt.api")

RETRIEVER_REGISTRY: Dict[str, ParameterizedRetriever] = {}
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting ResearchGPT API: initializing ParameterizedRetriever cache...")
    for col_name in ALL_COLLECTION_NAMES:
        try:
            RETRIEVER_REGISTRY[col_name] = ParameterizedRetriever(
                collection_name=col_name,
                persist_directory=str(CHROMA_DB_DIR),
            )
            logger.info(f"Initialized retriever for collection: {col_name}")
        except Exception as exc:
            logger.warning(f"Could not load collection '{col_name}': {exc}")
    yield
    RETRIEVER_REGISTRY.clear()


app = FastAPI(
    title="ResearchGPT Ablation & Retrieval API",
    description="Production FastAPI backend for benchmarking chunking strategies, Hybrid BM25+RRF retrieval, and Cross-Encoder reranking on arXiv literature.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_retrieval_output(raw_output: Any) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Handles both dict {'results': [...], 'timings': {...}} and tuple (candidates, timings) return conventions."""
    if isinstance(raw_output, dict):
        items = raw_output.get("results", [])
        timings = raw_output.get("timings", {})
        return items, timings
    elif isinstance(raw_output, tuple) and len(raw_output) == 2:
        return raw_output[0], raw_output[1]
    elif isinstance(raw_output, list):
        timings = raw_output[0].get("timings", {}) if raw_output and isinstance(raw_output[0], dict) else {}
        return raw_output, timings
    return [], {}


def _run_strategy(
    retriever: ParameterizedRetriever, query: str, strategy: str, top_k: int
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    t_start = time.perf_counter_ns()
    if strategy == "dense":
        raw = retriever.search_dense(query, top_k=top_k)
    elif strategy == "hybrid":
        raw = retriever.search_hybrid(query, top_k=top_k)
    elif strategy == "hybrid_rerank":
        raw = retriever.search_hybrid_rerank(query, top_k=top_k)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")
    elapsed_ms = (time.perf_counter_ns() - t_start) / 1e6

    items, timings = _normalize_retrieval_output(raw)
    if "total_ms" not in timings:
        timings["total_ms"] = round(elapsed_ms, 3)
    return items, timings


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "loaded_collections": list(RETRIEVER_REGISTRY.keys()),
        "groq_configured": bool(GROQ_API_KEY),
    }


@app.get("/collections")
async def get_collections():
    stats_csv = PROCESSED_DIR / "index_statistics.csv"
    index_stats = (
        pd.read_csv(stats_csv).to_dict(orient="records")
        if stats_csv.exists()
        else []
    )
    return {
        "available_collections": ALL_COLLECTION_NAMES,
        "loaded_collections": list(RETRIEVER_REGISTRY.keys()),
        "index_statistics": index_stats,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    retriever = RETRIEVER_REGISTRY.get(req.collection)
    if not retriever:
        if req.collection in ALL_COLLECTION_NAMES:
            try:
                retriever = ParameterizedRetriever(
                    collection_name=req.collection,
                    persist_directory=str(CHROMA_DB_DIR),
                )
                RETRIEVER_REGISTRY[req.collection] = retriever
            except Exception as e:
                logger.error(f"Lazy load failed for {req.collection}: {e}")

    if not retriever:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection}' is not loaded. Available: {list(RETRIEVER_REGISTRY.keys())}",
        )

    items, raw_timings = _run_strategy(retriever, req.query, req.strategy, req.top_k)

    chunks: List[RetrievedChunk] = []
    for item in items:
        score_val = (
            item.get("rerank_score")
            or item.get("reranker_score")
            or item.get("final_score")
            or item.get("rrf_score")
            or item.get("dense_score")
            or item.get("score")
            or item.get("similarity")
            or 0.0
        )
        chunks.append(
            RetrievedChunk(
                id=str(item.get("id", "")),
                document=str(item.get("document", "")),
                metadata=item.get("metadata", {}),
                score=float(score_val),
            )
        )

    answer = None
    gen_ms = 0.0
    if req.generate_answer:
        if not groq_client:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not configured on the server. Set generate_answer=false or configure GROQ_API_KEY.",
            )
        t0 = time.perf_counter_ns()
        context_text = "\n\n".join(
            f"[{i + 1}] (Paper ID: {c.metadata.get('paper_id', 'Unknown')}) {c.document}"
            for i, c in enumerate(chunks)
        )
        prompt = (
            "You are an academic AI assistant. Answer the research question strictly using "
            "the provided context passages. Cite passages inline using [1], [2], etc. "
            "If the context does not contain the answer, state that explicitly.\n\n"
            f"Context:\n{context_text[:8000]}\n\nQuestion: {req.query}"
        )
        completion = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        answer = completion.choices[0].message.content
        gen_ms = (time.perf_counter_ns() - t0) / 1e6

    retr_total = float(raw_timings.get("total_retrieval_ms", raw_timings.get("total_ms", 0.0)))
    timings = LatencyBreakdown(
        embedding_ms=float(raw_timings.get("query_embedding_ms", raw_timings.get("embedding_ms", 0.0))),
        dense_ms=float(raw_timings.get("dense_search_ms", raw_timings.get("dense_ms", 0.0))),
        sparse_bm25_ms=float(
            raw_timings.get("sparse_search_ms", raw_timings.get("bm25_ms", raw_timings.get("sparse_bm25_ms", 0.0)))
        ),
        rrf_fusion_ms=float(
            raw_timings.get("rrf_merge_ms", raw_timings.get("rrf_ms", raw_timings.get("rrf_fusion_ms", 0.0)))
        ),
        rerank_ms=float(raw_timings.get("rerank_ms", 0.0)),
        generation_ms=round(gen_ms, 3),
        total_ms=round(retr_total + gen_ms, 3),
    )

    return QueryResponse(
        query=req.query,
        collection=req.collection,
        strategy=req.strategy,
        top_k=req.top_k,
        answer=answer,
        chunks=chunks,
        timings=timings,
    )


@app.post("/ablation/compare")
async def compare_ablation_configs(req: CompareRequest):
    comparisons = []
    for col in req.collections:
        retriever = RETRIEVER_REGISTRY.get(col)
        if not retriever:
            try:
                retriever = ParameterizedRetriever(
                    collection_name=col,
                    persist_directory=str(CHROMA_DB_DIR),
                )
                RETRIEVER_REGISTRY[col] = retriever
            except Exception:
                continue
        for strat in req.strategies:
            items, timings = _run_strategy(retriever, req.query, strat, req.top_k)
            comparisons.append(
                {
                    "collection": col,
                    "strategy": strat,
                    "timings": timings,
                    "retrieved_paper_ids": [
                        item.get("metadata", {}).get("paper_id") for item in items
                    ],
                    "top_chunk_preview": (
                        items[0].get("document", "")[:240] + "..." if items else ""
                    ),
                }
            )
    return {"query": req.query, "top_k": req.top_k, "comparisons": comparisons}


@app.get("/benchmarks/summary")
async def get_benchmarks_summary():
    bench_path = PROCESSED_DIR / "benchmark_results.csv"
    sig_path = PROCESSED_DIR / "statistical_significance.csv"

    if not bench_path.exists():
        raise HTTPException(status_code=404, detail="benchmark_results.csv not found.")

    df = pd.read_csv(bench_path)
    group_cols = [
        c
        for c in ["collection", "strategy", "top_k", "Collection", "Strategy", "Cutoff", "TopK"]
        if c in df.columns
    ]
    numeric_cols = [
        c for c in df.select_dtypes(include="number").columns.tolist()
        if c not in group_cols
    ]

    aggregated = (
        df.groupby(group_cols)[numeric_cols]
        .mean()
        .round(4)
        .reset_index()
        .to_dict(orient="records")
        if group_cols
        else df.to_dict(orient="records")
    )

    significance = (
        pd.read_csv(sig_path).fillna(0.0).to_dict(orient="records")
        if sig_path.exists()
        else []
    )

    return {
        "total_benchmark_rows": len(df),
        "aggregated_metrics": aggregated,
        "statistical_significance": significance,
    }
