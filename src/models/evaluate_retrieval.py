try:
    import msvc_runtime
except ImportError:
    pass

import os
import json
import numpy as np
import pandas as pd
from advanced_retriever import AdvancedResearchRetriever

class RetrievalEvaluator:
    """
    Automated benchmarking harness to compare:
    1. Dense Only (ChromaDB Vector Search)
    2. Hybrid (Dense + BM25 Sparse)
    3. Hybrid + Cross-Encoder Reranking
    """
    def __init__(self, benchmark_path="data/evaluation/benchmark_queries.json"):
        print("Initializing Evaluation Framework...")
        if not os.path.exists(benchmark_path):
            raise FileNotFoundError(f"Benchmark file missing at: {benchmark_path}")
            
        with open(benchmark_path, "r", encoding="utf-8") as f:
            self.queries = json.load(f)
            
        self.retriever = AdvancedResearchRetriever()
        self.total_queries = len(self.queries)
        print(f"Loaded {self.total_queries} benchmark queries successfully.\n")

    def _is_relevant(self, document_text: str, expected_keywords: list, min_matches: int = 2) -> bool:
        """
        Relevance evaluation logic using keyword validation.
        A document is flagged relevant if it hits the minimum match threshold.
        """
        text_lower = document_text.lower()
        matches = sum(1 for kw in expected_keywords if kw.lower() in text_lower)
        return matches >= min_matches

    def evaluate_pipeline(self):
        # Tracker dictionaries for metrics across architectures
        metrics = {
            "Dense Only": {"p@5": [], "r@5": [], "r@10": [], "rr": []},
            "Hybrid": {"p@5": [], "r@5": [], "r@10": [], "rr": []},
            "Hybrid + Reranker": {"p@5": [], "r@5": [], "r@10": [], "rr": []}
        }

        print(f"Running evaluation loops over {self.total_queries} queries...")
        
        for idx, item in enumerate(self.queries, 1):
            query_text = item["query"]
            keywords = item["expected_keywords"]
            
            # Establish baseline target size within current corpus for Recall calculations
            all_corpus_texts = self.retriever.df["text"].tolist()
            total_relevant_in_corpus = max(1, sum(1 for text in all_corpus_texts if self._is_relevant(text, keywords)))

            # --- Test Stage 1: Dense Only ---
            dense_results = self.retriever.collection.query(query_texts=[query_text], n_results=10)
            dense_docs = dense_results["documents"][0]
            self._compute_query_metrics(dense_docs, keywords, total_relevant_in_corpus, metrics["Dense Only"])

            # --- Test Stage 2: Hybrid Only (Top 10 raw documents) ---
            hybrid_candidates = self.retriever.hybrid_search(query_text, top_k_dense=10, top_k_sparse=10)[:10]
            hybrid_docs = [doc["text"] for doc in hybrid_candidates]
            self._compute_query_metrics(hybrid_docs, keywords, total_relevant_in_corpus, metrics["Hybrid"])

            # --- Test Stage 3: Hybrid + Cross-Encoder Reranker (Top 10 reranked documents) ---
            reranked_candidates = self.retriever.retrieve_and_rerank(query_text, final_top_k=10)
            reranked_docs = [doc["text"] for doc in reranked_candidates]
            self._compute_query_metrics(reranked_docs, keywords, total_relevant_in_corpus, metrics["Hybrid + Reranker"])

        self._print_final_report(metrics)

    def _compute_query_metrics(self, retrieved_docs: list, keywords: list, total_rel: int, target_dict: dict):
        # Construct binary array representing hits/misses: e.g., [1, 0, 1, 1, 0]
        rel_array = [1 if self._is_relevant(doc, keywords) else 0 for doc in retrieved_docs]
        
        # Calculate Precision@5
        p_at_5 = sum(rel_array[:5]) / 5.0 if len(rel_array) >= 5 else (sum(rel_array) / float(len(rel_array)) if rel_array else 0.0)
        target_dict["p@5"].append(p_at_5)
        
        # Calculate Recall@5 and Recall@10
        r_at_5 = sum(rel_array[:5]) / float(total_rel)
        r_at_10 = sum(rel_array[:10]) / float(total_rel)
        target_dict["r@5"].append(r_at_5)
        target_dict["r@10"].append(r_at_10)
        
        # Calculate Reciprocal Rank (RR) for MRR
        rr = 0.0
        for position, value in enumerate(rel_array):
            if value == 1:
                rr = 1.0 / (position + 1)
                break
        target_dict["rr"].append(rr)

    def _print_final_report(self, metrics: dict):
        print("\n" + "="*75)
        print(" PERFORMANCE BENCHMARK REPORT: TWO-STAGE RETRIEVAL SPECTRUM")
        print("="*75)
        
        summary_table = []
        for architecture_name, score_data in metrics.items():
            summary_table.append({
                "Architecture Strategy": architecture_name,
                "Precision@5": round(np.mean(score_data["p@5"]), 4),
                "Recall@5": round(np.mean(score_data["r@5"]), 4),
                "Recall@10": round(np.mean(score_data["r@10"]), 4),
                "MRR": round(np.mean(score_data["rr"]), 4)
            })
            
        df_summary = pd.DataFrame(summary_table)
        print(df_summary.to_string(index=False))
        print("="*75)
        print("Metrics derived from automated query verification suite.")

if __name__ == "__main__":
    evaluator = RetrievalEvaluator()
    evaluator.evaluate_pipeline()