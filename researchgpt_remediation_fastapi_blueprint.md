# ResearchGPT: Codebase Remediation & FastAPI Backend Execution Plan

> **Target Agent Instructions (Antigravity):**  
> Execute the tasks in this specification sequentially from **Step 1** through **Step 9**. Do not skip steps. Verify file paths, preserve existing core logic in `ParameterizedRetriever` and chunking classes, and run the validation commands at the end of the execution.

---

## Step 1: Package Structure & Dead Code Cleanup (P0.3 & P1.2)

### 1.1 Create Python Package Initializers
Create empty `__init__.py` files in the following directories so all modules can be imported cleanly from the project root without `sys.path` hacks:
- `src/__init__.py`
- `src/ingestion/__init__.py`
- `src/models/__init__.py`
- `src/evaluation/__init__.py`
- `src/api/__init__.py` *(create the `src/api/` directory)*

### 1.2 Remove Duplicate & Dead Legacy Files
Delete the following obsolete files if they exist:
- `src/models/rag_pipeline.py` *(Dead Phase 1 single-collection code)*
- `src/data/download_pdfs.py` *(Duplicate of `src/ingestion/fetch_pdfs.py`)*

---

## Step 2: Update `requirements.txt` (P0.1)

Ensure `requirements.txt` contains all existing dependencies plus the missing retrieval, retry, and FastAPI packages. Append or merge the following entries into `requirements.txt`:

```text
rank-bm25>=0.2.2
requests>=2.31.0
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.9.0
python-dotenv>=1.0.1
tenacity>=9.0.0
pytest>=8.0.0
httpx>=0.27.0
```

---

## Step 3: Cross-Platform `msvc_runtime` Guard (P0.2)

Inspect the following 5 files:
1. `src/ingestion/chunkers.py`
2. `src/ingestion/profiler.py`
3. `src/models/build_embeddings.py`
4. `src/models/index_profiler.py`
5. `src/evaluation/plot_chunks.py`

In each file, replace the bare `import msvc_runtime` statement at the top of the file with a safe cross-platform `try/except` block:

```python
try:
    import msvc_runtime
except ImportError:
    pass
```

---

## Step 4: Centralized Configuration Module (P1.2)

Create `src/config.py` to eliminate hardcoded paths, collection names, and model identifiers across the repository.

### File: `src/config.py`
```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
EVALUATION_DIR = DATA_DIR / "evaluation"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

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
```

---

## Step 5: Fix Broken Imports in `generate_answer.py` & `auto_label_benchmark.py` (P0.3 & P0.5)

Both `src/models/generate_answer.py` and `src/evaluation/auto_label_benchmark.py` fail on import because they reference `AdvancedResearchRetriever`, which was replaced by `ParameterizedRetriever` in `src/models/advanced_retriever.py`.

### 5.1 Update `src/models/generate_answer.py`
Refactor `src/models/generate_answer.py` to use `ParameterizedRetriever` and `src.config`:

```python
import os
import sys
from groq import Groq
from src.config import CHROMA_DB_DIR, COLLECTIONS, GROQ_API_KEY, LLM_MODEL
from src.models.advanced_retriever import ParameterizedRetriever


def generate_research_answer(
    query: str,
    collection_name: str = COLLECTIONS["semantic"],
    strategy: str = "hybrid_rerank",
    top_k: int = 5,
) -> dict:
    retriever = ParameterizedRetriever(
        collection_name=collection_name,
        persist_directory=str(CHROMA_DB_DIR),
    )

    if strategy == "dense":
        retrieval_out = retriever.search_dense(query, top_k=top_k)
    elif strategy == "hybrid":
        retrieval_out = retriever.search_hybrid(query, top_k=top_k)
    else:
        retrieval_out = retriever.search_hybrid_rerank(query, top_k=top_k)

    results = retrieval_out.get("results", retrieval_out) if isinstance(retrieval_out, dict) else retrieval_out
    timings = retrieval_out.get("timings", {}) if isinstance(retrieval_out, dict) else {}

    context_blocks = []
    for idx, item in enumerate(results):
        meta = item.get("metadata", {})
        paper_id = meta.get("paper_id", "Unknown")
        context_blocks.append(f"[{idx + 1}] (Paper ID: {paper_id})\n{item.get('document', '')}")

    context_str = "\n\n".join(context_blocks)[:8000]

    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY environment variable is not set.")

    client = Groq(api_key=GROQ_API_KEY)
    prompt = (
        "You are an academic AI research assistant. Answer the query strictly using the provided context. "
        "Cite sources inline using bracketed numbers like [1], [2]. If the context is insufficient, state so clearly.\n\n"
        f"Context:\n{context_str}\n\nQuery: {query}"
    )

    completion = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )

    return {
        "query": query,
        "answer": completion.choices[0].message.content,
        "results": results,
        "timings": timings,
    }


if __name__ == "__main__":
    test_q = sys.argv[1] if len(sys.argv) > 1 else "What are the main challenges in LLM unlearning?"
    output = generate_research_answer(test_q)
    print("\n--- Answer ---\n", output["answer"])
```

### 5.2 Update `src/evaluation/auto_label_benchmark.py`
Refactor `src/evaluation/auto_label_benchmark.py` so it imports `ParameterizedRetriever` instead of `AdvancedResearchRetriever`, reads candidate queries from `data/evaluation/benchmark_queries.json`, pools top chunks across strategies, and uses Groq to assign silver-standard `relevant_paper_ids` to expand `data/benchmark_queries.json` to $\ge 25$ queries.

```python
import json
import logging
from groq import Groq
from tenacity import retry, wait_exponential, stop_after_attempt
from src.config import CHROMA_DB_DIR, COLLECTIONS, DATA_DIR, EVALUATION_DIR, GROQ_API_KEY, LLM_MODEL
from src.models.advanced_retriever import ParameterizedRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=3, max=30))
def judge_relevance(client: Groq, query: str, chunk_text: str) -> bool:
    prompt = (
        "Determine if the following document excerpt contains specific technical information "
        "relevant to answering the research query. Respond ONLY with a JSON object: "
        '{"relevant": true} or {"relevant": false}.\n\n'
        f"Query: {query}\nExcerpt: {chunk_text[:2500]}"
    )
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    payload = json.loads(resp.choices[0].message.content or "{}")
    return bool(payload.get("relevant", False))


def run_silver_labeling():
    raw_queries_path = EVALUATION_DIR / "benchmark_queries.json"
    output_path = DATA_DIR / "benchmark_queries.json"

    if not raw_queries_path.exists():
        logger.error(f"Source queries not found at {raw_queries_path}")
        return

    with open(raw_queries_path, "r", encoding="utf-8") as f:
        candidate_queries = json.load(f)

    retriever = ParameterizedRetriever(
        collection_name=COLLECTIONS["fixed"],
        persist_directory=str(CHROMA_DB_DIR),
    )
    client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

    labeled_queries = []
    for idx, item in enumerate(candidate_queries):
        q_id = item.get("query_id") or item.get("id") or f"q_{idx + 1:02d}"
        q_text = item.get("query", "")
        existing_rel = item.get("relevant_paper_ids", [])

        if existing_rel:
            labeled_queries.append({
                "query_id": q_id,
                "query": q_text,
                "relevant_paper_ids": existing_rel,
            })
            continue

        res = retriever.search_hybrid_rerank(q_text, top_k=5)
        chunks = res.get("results", res) if isinstance(res, dict) else res

        relevant_papers = set()
        for chunk in chunks:
            paper_id = chunk.get("metadata", {}).get("paper_id")
            if not paper_id:
                continue
            if client:
                try:
                    if judge_relevance(client, q_text, chunk.get("document", "")):
                        relevant_papers.add(paper_id)
                except Exception as e:
                    logger.warning(f"LLM labeling fallback for {q_id}: {e}")
            else:
                # Top-1 fallback if no API key provided
                relevant_papers.add(paper_id)
                break

        if not relevant_papers and chunks:
            top_pid = chunks[0].get("metadata", {}).get("paper_id")
            if top_pid:
                relevant_papers.add(top_pid)

        labeled_queries.append({
            "query_id": q_id,
            "query": q_text,
            "relevant_paper_ids": sorted(list(relevant_papers)),
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(labeled_queries, f, indent=2)
    logger.info(f"Saved {len(labeled_queries)} labeled benchmark queries to {output_path}")


if __name__ == "__main__":
    run_silver_labeling()
```

---

## Step 6: Fix LLM-as-a-Judge Silent Failures in `evaluate_retrieval.py` (P0.4)

In `src/models/evaluate_retrieval.py`, `CitationSupportRate` and `UnsupportedClaimRate` previously returned `0.0` across all rows because exceptions from Groq API rate limits were silently swallowed and JSON parsing was fragile.

Update `generate_answer()` and `evaluate_generation()` in `src/models/evaluate_retrieval.py` to use `tenacity` exponential backoff, structured JSON mode, and explicit error logging:

```python
import json
import logging
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import Groq, RateLimitError, APIError
from src.config import GROQ_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError)),
)
def _call_groq_with_backoff(messages: list, json_mode: bool = False) -> str:
    if not groq_client:
        raise RuntimeError("GROQ_API_KEY is not configured.")
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.0,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = groq_client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def generate_answer(query: str, context: str) -> str:
    if not groq_client:
        logger.warning("Skipping generation: GROQ_API_KEY not set.")
        return ""
    prompt = (
        "Answer the research question strictly based on the context below. "
        "Cite passages using [1], [2], etc.\n\n"
        f"Context:\n{context[:8000]}\n\nQuestion: {query}"
    )
    try:
        return _call_groq_with_backoff([{"role": "user", "content": prompt}], json_mode=False)
    except Exception as e:
        logger.error(f"generate_answer failed for query '{query[:40]}...': {e}")
        return ""


def evaluate_generation(query: str, context: str, generated_answer: str) -> tuple[float, float]:
    if not generated_answer or not generated_answer.strip():
        logger.warning(f"Empty generated_answer for query '{query[:40]}...'; returning (0.0, 0.0)")
        return 0.0, 0.0

    prompt = f"""You are an impartial academic evaluation judge. Evaluate the Generated Answer against the Retrieved Context.
Return ONLY a valid JSON object with two keys:
- "citation_support_rate": float in [0.0, 1.0] indicating the proportion of claims supported by the context.
- "unsupported_claim_rate": float in [0.0, 1.0] indicating the proportion of hallucinated or unsupported claims.

Query: {query}
Retrieved Context: {context[:8000]}
Generated Answer: {generated_answer}"""

    try:
        raw_response = _call_groq_with_backoff(
            [{"role": "user", "content": prompt}],
            json_mode=True,
        )
        parsed = json.loads(raw_response)
        csr = float(parsed.get("citation_support_rate", 0.0))
        ucr = float(parsed.get("unsupported_claim_rate", 0.0))
        return max(0.0, min(1.0, csr)), max(0.0, min(1.0, ucr))
    except Exception as e:
        logger.error(f"evaluate_generation failed for query '{query[:40]}...': {e}")
        return 0.0, 0.0
```

---

## Step 7: Make Benchmark Plots Data-Driven + Add Cohen's d (P1.1 & P2.7)

### 7.1 Refactor `src/evaluation/plot_benchmark_results.py`
Replace all hardcoded metric arrays in `src/evaluation/plot_benchmark_results.py` so that every plot dynamically loads and aggregates `data/processed/benchmark_results.csv`.

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.config import PROCESSED_DIR

PLOTS_DIR = PROCESSED_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def load_benchmark_df() -> pd.DataFrame:
    csv_path = PROCESSED_DIR / "benchmark_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Cannot generate plots; {csv_path} does not exist.")
    return pd.read_csv(csv_path)


def plot_ndcg_by_strategy(df: pd.DataFrame):
    col_metric = "ndcg_at_k" if "ndcg_at_k" in df.columns else "nDCG@K"
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_chunk = "collection" if "collection" in df.columns else "Collection"

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x=col_chunk, y=col_metric, hue=col_strat, errorbar=("ci", 95))
    plt.title("nDCG@K Across Chunking Collections & Retrieval Strategies")
    plt.ylabel("Mean nDCG@K")
    plt.xlabel("Collection")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ndcg_by_strategy.png", dpi=300)
    plt.close()


def plot_latency_breakdown(df: pd.DataFrame):
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_lat = "total_latency_ms" if "total_latency_ms" in df.columns else "TotalLatency_ms"

    if col_lat not in df.columns:
        return

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x=col_strat, y=col_lat)
    plt.yscale("log")
    plt.title("Retrieval Latency Distribution by Strategy (Log Scale)")
    plt.ylabel("Latency (ms)")
    plt.xlabel("Retrieval Strategy")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "latency_breakdown.png", dpi=300)
    plt.close()


def plot_precision_recall_tradeoff(df: pd.DataFrame):
    col_p = "precision_at_k" if "precision_at_k" in df.columns else "Precision@K"
    col_r = "recall_at_k" if "recall_at_k" in df.columns else "Recall@K"
    col_strat = "strategy" if "strategy" in df.columns else "Strategy"
    col_chunk = "collection" if "collection" in df.columns else "Collection"

    agg = df.groupby([col_chunk, col_strat])[[col_p, col_r]].mean().reset_index()
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=agg, x=col_r, y=col_p, hue=col_chunk, style=col_strat, s=140)
    plt.title("Precision@K vs. Recall@K Across Configurations")
    plt.xlabel("Mean Recall@K")
    plt.ylabel("Mean Precision@K")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "precision_recall_tradeoff.png", dpi=300)
    plt.close()


def main():
    df = load_benchmark_df()
    plot_ndcg_by_strategy(df)
    plot_latency_breakdown(df)
    plot_precision_recall_tradeoff(df)
    print(f"Saved data-driven benchmark plots to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
```

### 7.2 Add Cohen's $d$ Effect Size to `src/evaluation/statistical_tests.py`
In `src/evaluation/statistical_tests.py`, add a helper function for paired Cohen's $d$ and include `cohens_d` as a column in `statistical_significance.csv`:

```python
import numpy as np

def compute_paired_cohens_d(x: np.ndarray, y: np.ndarray) -> float:
    diff = np.asarray(x, dtype=float) - np.asarray(y, dtype=float)
    std_diff = np.std(diff, ddof=1)
    if std_diff == 0 or np.isnan(std_diff):
        return 0.0
    return float(np.mean(diff) / std_diff)
```

---

## Step 8: Build the FastAPI Backend (`src/api/`) (P2.3)

Create the FastAPI application with an asynchronous `lifespan` manager that pre-loads and caches `ParameterizedRetriever` instances for all 4 ChromaDB collections on startup so queries execute with sub-25ms retrieval latency.

### 8.1 File: `src/api/schemas.py`
```python
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
```

### 8.2 File: `src/api/app.py`
```python
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
    """Handles both dict {'results': [...], 'timings': {...}} and list return conventions."""
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
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{req.collection}' is not loaded. Available: {list(RETRIEVER_REGISTRY.keys())}",
        )

    items, raw_timings = _run_strategy(retriever, req.query, req.strategy, req.top_k)

    chunks: List[RetrievedChunk] = []
    for item in items:
        score_val = (
            item.get("rerank_score")
            or item.get("rrf_score")
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

    retr_total = float(raw_timings.get("total_ms", 0.0))
    timings = LatencyBreakdown(
        embedding_ms=float(raw_timings.get("embedding_ms", 0.0)),
        dense_ms=float(raw_timings.get("dense_ms", 0.0)),
        sparse_bm25_ms=float(raw_timings.get("bm25_ms", raw_timings.get("sparse_bm25_ms", 0.0))),
        rrf_fusion_ms=float(raw_timings.get("rrf_ms", raw_timings.get("rrf_fusion_ms", 0.0))),
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
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    group_cols = [c for c in ["collection", "strategy", "top_k", "Collection", "Strategy", "TopK"] if c in df.columns]

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
```

---

## Step 9: Automated Pytest Suite (`tests/`) (P1.4)

Create a `tests/` directory at the project root with unit and API tests so `pytest` validates the codebase automatically.

### 9.1 File: `tests/test_api_and_core.py`
```python
from fastapi.testclient import TestClient
from src.api.app import app
from src.config import ALL_COLLECTION_NAMES, STRATEGIES


client = TestClient(app)


def test_config_constants():
    assert len(ALL_COLLECTION_NAMES) == 4
    assert "arxiv_semantic_512" in ALL_COLLECTION_NAMES
    assert "hybrid_rerank" in STRATEGIES


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "loaded_collections" in data


def test_collections_endpoint():
    response = client.get("/collections")
    assert response.status_code == 200
    data = response.json()
    assert "available_collections" in data
    assert len(data["available_collections"]) == 4


def test_query_validation_error():
    # Query shorter than min_length=3 should return 422 Unprocessable Entity
    response = client.post(
        "/query",
        json={
            "query": "hi",
            "collection": "arxiv_semantic_512",
            "strategy": "dense",
            "top_k": 3,
            "generate_answer": False,
        },
    )
    assert response.status_code == 422
```

---

## Step 10: Agent Verification Checklist

After applying Steps 1–9, execute the following verification commands in the terminal from the project root:

```bash
# 1. Install updated dependencies
pip install -r requirements.txt

# 2. Verify no broken imports across all modules
python -c "import src.config; import src.models.advanced_retriever; import src.models.generate_answer; import src.evaluation.auto_label_benchmark; import src.api.app"

# 3. Run unit & FastAPI test suite
pytest tests/ -v

# 4. Regenerate data-driven plots
python -m src.evaluation.plot_benchmark_results

# 5. Start the FastAPI server and verify OpenAPI schema
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```