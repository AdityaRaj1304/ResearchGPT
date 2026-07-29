try:
    import msvc_runtime
except ImportError:
    pass

import pandas as pd
from advanced_retriever import ParameterizedRetriever

def run_tests():
    queries = [
        "How do transformer attention mechanisms scale with sequence length?",
        "What optimization techniques prevent gradient explosion during training?",
        "What baseline benchmarks were used to evaluate model accuracy?",
        "How are positional encodings integrated into multi-head attention blocks?"
    ]
    
    collections = [
        "arxiv_wholedoc",
        "arxiv_fixed_512",
        "arxiv_overlap_512_128",
        "arxiv_semantic"
    ]
    
    strategies = ["dense", "hybrid", "hybrid_rerank"]
    
    results = []
    
    for coll in collections:
        retriever = ParameterizedRetriever(collection_name=coll)
        for strategy in strategies:
            for q_idx, query in enumerate(queries):
                try:
                    if strategy == "dense":
                        docs, timings = retriever.search_dense(query, top_k=3)
                    elif strategy == "hybrid":
                        docs, timings = retriever.search_hybrid(query, top_k=3)
                    else:
                        docs, timings = retriever.search_hybrid_rerank(query, top_k=3)
                        
                    assert timings["query_embedding_ms"] > 0, "Query embedding time isolated check failed."
                    if strategy != "hybrid_rerank":
                        assert timings["rerank_ms"] == 0.0, f"rerank_ms not 0.0 for {strategy}."
                    
                    if strategy == "hybrid_rerank":
                        for d in docs:
                            assert d["rrf_score"] is not None and d["reranker_score"] is not None, "Both rrf_score and reranker_score must exist for reranked runs."
                    
                    results.append({
                        "Collection": coll,
                        "Strategy": strategy,
                        "Q": f"Q{q_idx + 1}",
                        "Embed (ms)": timings["query_embedding_ms"],
                        "Dense (ms)": timings["dense_search_ms"],
                        "Sparse (ms)": timings["sparse_search_ms"],
                        "RRF (ms)": timings["rrf_merge_ms"],
                        "Rerank (ms)": timings["rerank_ms"],
                        "Total (ms)": timings["total_retrieval_ms"]
                    })
                except Exception as e:
                    print(f"FAILED on {coll} | {strategy} | Q{q_idx+1}: {e}")
                    raise
                    
    df = pd.DataFrame(results)
    
    for col in ["Embed (ms)", "Dense (ms)", "Sparse (ms)", "RRF (ms)", "Rerank (ms)", "Total (ms)"]:
        df[col] = df[col].apply(lambda x: f"{x:.2f}")
        
    print("\n" + "="*90)
    print("PHASE 4: HYBRID RETRIEVAL & LATENCY PROFILING MATRIX")
    print("="*90)
    print(df.to_markdown(index=False))
    print("="*90)
    print("100% execution pass rate across all 48 test runs achieved.")

if __name__ == "__main__":
    run_tests()
