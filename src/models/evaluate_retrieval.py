try:
    import msvc_runtime
except ImportError:
    pass

import os
import json
import time
import sys
import re
import numpy as np
import pandas as pd
from openai import OpenAI

from advanced_retriever import ParameterizedRetriever

COLLECTIONS = ["arxiv_wholedoc", "arxiv_fixed_512", "arxiv_overlap_512_128", "arxiv_semantic"]
STRATEGIES = ["dense", "hybrid", "hybrid_rerank"]
CUTOFFS = [3, 5]

def normalize_paper_id(paper_id_str: str) -> str:
    if not paper_id_str:
        return ""
    # Convert to string and lowercase
    s = str(paper_id_str).strip().lower()
    # Strip file extensions and URLs
    s = s.replace(".pdf", "").split("/")[-1]
    # Extract standard arXiv ID pattern (e.g., '2401.12345' from '2401.12345v1')
    match = re.search(r'\d{4}\.\d{4,5}', s)
    if match:
        return match.group(0)
    return s

class BenchmarkEvaluator:
    def __init__(self, benchmark_path="data/benchmark_queries.json"):
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Missing {benchmark_path}")
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)
            
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("\n" + "!"*60)
            print("CRITICAL ERROR: GROQ_API_KEY not found in environment variables.")
            print("Please export your API key to run the LLM-as-a-judge evaluation.")
            print("Example (PowerShell): $env:GROQ_API_KEY='gsk_...'")
            print("!"*60 + "\n")
            sys.exit(1)
            
        self.llm_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=api_key)
        self.model_name = "llama-3.3-70b-versatile"

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
        context_block = "\n---\n".join([f"[Paper {d['metadata'].get('paper_id', 'Unknown')}] {d['document']}" for d in docs])
        # Truncate context to avoid Groq TPM limits (12k tokens per minute)
        if len(context_block) > 8000:
            context_block = context_block[:8000] + "... [TRUNCATED]"
            
        sys_prompt = "You are a research assistant. Answer the query using ONLY the provided literature. Cite sources using [Paper X]."
        user_prompt = f"QUERY: {query}\n\nCONTEXT:\n{context_block}"
        
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0
            )
            time.sleep(5) # Avoid TPM limit
            return resp.choices[0].message.content
        except Exception as e:
            print(f"Generation error: {repr(e)}")
            return ""

    def evaluate_generation(self, query: str, context: list, answer: str):
        if not answer:
            return 0.0, 0.0
            
        sys_prompt = """You are an objective judge. Given a query, a set of provided context documents, and a generated answer, you must determine:
1. Citation Support Rate: What percentage of the substantive claims in the answer are supported by the context?
2. Unsupported Claim Rate: What percentage of claims are hallucinations (not in context)?

Return ONLY a JSON object: {"citation_support_rate": 0.0 to 100.0, "unsupported_claim_rate": 0.0 to 100.0}"""

        context_block = "\n---\n".join([f"[Paper {d['metadata'].get('paper_id', 'Unknown')}] {d['document']}" for d in context])
        if len(context_block) > 8000:
            context_block = context_block[:8000] + "... [TRUNCATED]"
            
        user_prompt = f"QUERY: {query}\n\nCONTEXT: {context_block}\n\nANSWER: {answer}"
        
        try:
            resp = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            res_json = json.loads(resp.choices[0].message.content)
            time.sleep(5)
            return float(res_json.get("citation_support_rate", 0)), float(res_json.get("unsupported_claim_rate", 0))
        except Exception as e:
            print(f"Judge error: {repr(e)}")
            return 0.0, 0.0

    def run(self):
        results = []
        for coll in COLLECTIONS:
            retriever = ParameterizedRetriever(collection_name=coll)
            for strat in STRATEGIES:
                print(f"Evaluating {coll} | {strat}")
                
                for q in self.queries:
                    q_text = q["query"]
                    rel_ids = q["relevant_paper_ids"]
                    diff = q["difficulty"]
                    topic = q["topic"]
                    
                    if strat == "dense":
                        docs, timings = retriever.search_dense(q_text, top_k=10)
                    elif strat == "hybrid":
                        docs, timings = retriever.search_hybrid(q_text, top_k=10)
                    else:
                        docs, timings = retriever.search_hybrid_rerank(q_text, top_k=10)
                        
                    relevant_ids_normalized = {normalize_paper_id(pid) for pid in rel_ids}
                    
                    rel_array = [1 if normalize_paper_id(d["metadata"].get("paper_id")) in relevant_ids_normalized else 0 for d in docs]
                    total_rel_papers = len(relevant_ids_normalized)
                    
                    # Recall@10
                    retrieved_rel_papers_10 = len({normalize_paper_id(d["metadata"].get("paper_id")) for d in docs if normalize_paper_id(d["metadata"].get("paper_id")) in relevant_ids_normalized})
                    r_10 = retrieved_rel_papers_10 / total_rel_papers if total_rel_papers > 0 else 0
                    
                    for k in CUTOFFS:
                        k_docs = docs[:k]
                        k_rel_array = rel_array[:k]
                        
                        p_k = sum(k_rel_array) / k if k > 0 else 0
                        retrieved_rel_papers_k = len({normalize_paper_id(d["metadata"].get("paper_id")) for d in k_docs if normalize_paper_id(d["metadata"].get("paper_id")) in relevant_ids_normalized})
                        r_k = retrieved_rel_papers_k / total_rel_papers if total_rel_papers > 0 else 0
                        
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
                            "QueryID": q["query_id"],
                            "Topic": topic,
                            "Difficulty": diff,
                            "Precision": p_k,
                            "Recall": r_k,
                            "Recall_10": r_10,
                            "MRR": mrr,
                            "nDCG": ndcg_k,
                            "CitationSupportRate": cit_sup,
                            "UnsupportedClaimRate": unsupp,
                            "TotalRetrievalMS": timings["total_retrieval_ms"]
                        })
                        
        df = pd.DataFrame(results)
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