try:
    import msvc_runtime
except ImportError:
    pass

import os
import json
import time
import sys
import re
import logging
import numpy as np
import pandas as pd
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from groq import Groq, RateLimitError, APIError

try:
    from src.config import GROQ_API_KEY, LLM_MODEL, PROCESSED_DIR, DATA_DIR, COLLECTIONS as CONFIG_COLLECTIONS, STRATEGIES as CONFIG_STRATEGIES
    from src.models.advanced_retriever import ParameterizedRetriever
except ImportError:
    from advanced_retriever import ParameterizedRetriever
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    PROCESSED_DIR = None
    DATA_DIR = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

COLLECTIONS = ["arxiv_wholedoc", "arxiv_fixed_512", "arxiv_overlap_512_128", "arxiv_semantic"]
STRATEGIES = ["dense", "hybrid", "hybrid_rerank"]
CUTOFFS = [3, 5]


def normalize_paper_id(paper_id_str: str) -> str:
    if not paper_id_str:
        return ""
    s = str(paper_id_str).strip().lower()
    s = s.replace(".pdf", "").split("/")[-1]
    match = re.search(r'\d{4}\.\d{4,5}', s)
    if match:
        return match.group(0)
    return s


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


class BenchmarkEvaluator:
    def __init__(self, benchmark_path="data/benchmark_queries.json"):
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Missing {benchmark_path}")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)
            
        if not GROQ_API_KEY:
            logger.warning(
                "GROQ_API_KEY not found in environment variables. "
                "Retrieval evaluation will run, but LLM generation/judge will be skipped."
            )

    def dcg_at_k(self, r, k):
        r = np.asarray(r, dtype=float)[:k]
        if r.size:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.

    def ndcg_at_k(self, r, k):
        dcg_max = self.dcg_at_k(sorted(r, reverse=True), k)
        if not dcg_max:
            return 0.
        return self.dcg_at_k(r, k) / dcg_max

    def generate_answer(self, query: str, docs: list) -> str:
        if isinstance(docs, list):
            context_block = "\n---\n".join(
                [f"[Paper {d.get('metadata', {}).get('paper_id', 'Unknown')}] {d.get('document', '')}" for d in docs]
            )
        else:
            context_block = str(docs)
        return generate_answer(query, context_block)

    def evaluate_generation(self, query: str, context: list, answer: str) -> tuple[float, float]:
        if isinstance(context, list):
            context_block = "\n---\n".join(
                [f"[Paper {d.get('metadata', {}).get('paper_id', 'Unknown')}] {d.get('document', '')}" for d in context]
            )
        else:
            context_block = str(context)
        return evaluate_generation(query, context_block, answer)

    def run(self):
        results = []
        for coll in COLLECTIONS:
            retriever = ParameterizedRetriever(collection_name=coll)
            for strat in STRATEGIES:
                print(f"Evaluating {coll} | {strat}")
                
                for q in self.queries:
                    q_text = q["query"]
                    rel_ids = q.get("relevant_paper_ids", [])
                    diff = q.get("difficulty", "medium")
                    topic = q.get("topic", "general")
                    
                    if strat == "dense":
                        retrieval_out = retriever.search_dense(q_text, top_k=10)
                    elif strat == "hybrid":
                        retrieval_out = retriever.search_hybrid(q_text, top_k=10)
                    else:
                        retrieval_out = retriever.search_hybrid_rerank(q_text, top_k=10)

                    if isinstance(retrieval_out, tuple) and len(retrieval_out) == 2:
                        docs, timings = retrieval_out
                    elif isinstance(retrieval_out, dict):
                        docs = retrieval_out.get("results", [])
                        timings = retrieval_out.get("timings", {})
                    else:
                        docs = retrieval_out if isinstance(retrieval_out, list) else []
                        timings = {}
                        
                    relevant_ids_normalized = {normalize_paper_id(pid) for pid in rel_ids}
                    
                    rel_array = [1 if normalize_paper_id(d.get("metadata", {}).get("paper_id")) in relevant_ids_normalized else 0 for d in docs]
                    total_rel_papers = len(relevant_ids_normalized)
                    
                    # Recall@10
                    retrieved_rel_papers_10 = len({normalize_paper_id(d.get("metadata", {}).get("paper_id")) for d in docs if normalize_paper_id(d.get("metadata", {}).get("paper_id")) in relevant_ids_normalized})
                    r_10 = retrieved_rel_papers_10 / total_rel_papers if total_rel_papers > 0 else 0.0
                    
                    for k in CUTOFFS:
                        k_docs = docs[:k]
                        k_rel_array = rel_array[:k]
                        
                        p_k = sum(k_rel_array) / k if k > 0 else 0.0
                        retrieved_rel_papers_k = len({normalize_paper_id(d.get("metadata", {}).get("paper_id")) for d in k_docs if normalize_paper_id(d.get("metadata", {}).get("paper_id")) in relevant_ids_normalized})
                        r_k = retrieved_rel_papers_k / total_rel_papers if total_rel_papers > 0 else 0.0
                        
                        mrr = 0.0
                        for pos, val in enumerate(k_rel_array):
                            if val == 1:
                                mrr = 1.0 / (pos + 1)
                                break
                                
                        ndcg_k = self.ndcg_at_k(k_rel_array, k)
                        
                        ans = self.generate_answer(q_text, k_docs)
                        cit_sup, unsupp = self.evaluate_generation(q_text, k_docs, ans)
                        
                        results.append({
                            "Collection": coll,
                            "Strategy": strat,
                            "Cutoff": k,
                            "QueryID": q.get("query_id", q.get("id", "Unknown")),
                            "Topic": topic,
                            "Difficulty": diff,
                            "Precision": p_k,
                            "Recall": r_k,
                            "Recall_10": r_10,
                            "MRR": mrr,
                            "nDCG": ndcg_k,
                            "CitationSupportRate": cit_sup,
                            "UnsupportedClaimRate": unsupp,
                            "TotalRetrievalMS": timings.get("total_retrieval_ms", timings.get("total_ms", 0.0))
                        })
                        
        df = pd.DataFrame(results)
        os.makedirs("data/processed", exist_ok=True)
        df.to_csv("data/processed/benchmark_results.csv", index=False)
        df.to_json("data/processed/benchmark_results.json", orient="records", indent=2)
        
        # Calculate Confidence Intervals
        stats_data = []
        for (coll, strat, cut), group in df.groupby(["Collection", "Strategy", "Cutoff"]):
            for metric in ["Precision", "Recall", "Recall_10", "MRR", "nDCG", "CitationSupportRate", "UnsupportedClaimRate"]:
                mean = group[metric].mean()
                sem = group[metric].sem() if len(group) > 1 else 0
                ci_95 = 1.96 * sem
                stats_data.append({
                    "Collection": coll,
                    "Strategy": strat,
                    "Cutoff": cut,
                    "Metric": metric,
                    "Mean": mean,
                    "SE": sem,
                    "CI_95": ci_95
                })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df.to_csv("data/processed/benchmark_stats.csv", index=False)
        print("Evaluation complete. Saved to data/processed/benchmark_results.csv")


if __name__ == "__main__":
    evaluator = BenchmarkEvaluator()
    evaluator.run()