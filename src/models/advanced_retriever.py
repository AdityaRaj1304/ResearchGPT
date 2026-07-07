# Self-healing Windows DLL fix (must be before any PyTorch / transformers imports!)
try:
    import msvc_runtime
except ImportError:
    pass

import os
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

class AdvancedResearchRetriever:
    """
    Phase 2 Implementation: Combines Dense Vector Search (ChromaDB ONNX) 
    with Sparse Keyword Search (BM25) and Cross-Encoder Reranking.
    """
    def __init__(self, input_parquet="data/processed/arxiv_dataset.parquet", db_path="data/vector_store", collection_name="arxiv_papers"):
        print("Initializing Advanced Phase 2 Hybrid Retrieval Engine...")
        
        # 1. Load Parquet Data for BM25 Keyword Indexing
        if not os.path.exists(input_parquet):
            raise FileNotFoundError(f"Cannot find {input_parquet}. Please verify your data directory.")
        self.df = pd.read_parquet(input_parquet)
        self.df["text"] = "Title: " + self.df["title"] + " | Abstract: " + self.df["abstract"]
        
        print("Building BM25 Sparse Keyword Index in memory...")
        # Tokenize text by lowercase word splitting for BM25
        tokenized_corpus = [doc.lower().split(" ") for doc in self.df["text"].tolist()]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Connect to ChromaDB Dense Vector Store
        print(f"Connecting to ChromaDB Vector Store at {db_path}...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.onnx_ef = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_collection(name=collection_name, embedding_function=self.onnx_ef)
        
        # 3. Initialize Cross-Encoder for Precision Reranking (Class 7b Homework Challenge 1)
        print("Loading MS-MARCO Cross-Encoder for deep reranking...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        print("Advanced Retrieval Engine fully armed and operational!\n" + "="*60)

    def hybrid_search(self, query: str, top_k_dense: int = 15, top_k_sparse: int = 15) -> list:
        """
        Executes dense vector search and sparse BM25 search simultaneously,
        then merges and deduplicates the candidate pool.
        """
        # --- Stage 1A: Dense Semantic Search ---
        dense_results = self.collection.query(query_texts=[query], n_results=top_k_dense)
        dense_ids = dense_results["ids"][0]
        
        # --- Stage 1B: Sparse BM25 Keyword Search ---
        tokenized_query = query.lower().split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top indices from BM25 scores
        top_sparse_indices = np.argsort(bm25_scores)[::-1][:top_k_sparse]
        sparse_ids = [self.df.iloc[idx]["id"] for idx in top_sparse_indices]
        
        # --- Stage 1C: Merge & Deduplicate ---
        combined_ids = list(set(dense_ids + sparse_ids))
        print(f"Hybrid Search retrieved {len(dense_ids)} dense + {len(sparse_ids)} sparse candidates -> {len(combined_ids)} unique papers.")
        
        # Fetch complete payloads for combined IDs
        candidate_docs = []
        for doc_id in combined_ids:
            row = self.df[self.df["id"] == doc_id].iloc[0]
            candidate_docs.append({
                "id": row["id"],
                "title": row["title"],
                "categories": row["categories"],
                "pdf_url": row["pdf_url"],
                "text": row["text"]
            })
            
        return candidate_docs

    def retrieve_and_rerank(self, query: str, final_top_k: int = 4) -> list:
        """
        Two-Stage Pipeline:
        1. High-Recall Hybrid Search (grabs ~25 unique candidates)
        2. High-Precision Cross-Encoder Reranking (narrows down to top 4)
        """
        # Step 1: Broad Hybrid Search
        candidates = self.hybrid_search(query)
        
        # Step 2: Prepare pairs for the Cross-Encoder [[query, doc_1], [query, doc_2], ...]
        print(f"Running deep Cross-Encoder self-attention over {len(candidates)} candidate pairs...")
        model_inputs = [[query, doc["text"]] for doc in candidates]
        
        # Step 3: Predict relevance logits
        rerank_scores = self.reranker.predict(model_inputs)
        
        # Attach scores to candidates and sort descending
        for idx, score in enumerate(rerank_scores):
            candidates[idx]["rerank_score"] = float(score)
            
        sorted_candidates = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
        return sorted_candidates[:final_top_k]

if __name__ == "__main__":
    # Test your Phase 2 Advanced RAG Retriever!
    retriever = AdvancedResearchRetriever()
    
    test_query = "what is rag ?"
    print(f"\nQUERY: '{test_query}'\n")
    
    top_papers = retriever.retrieve_and_rerank(query=test_query, final_top_k=4)
    
    print("\n--- FINAL RERANKED TOP PAPERS FOR LLM CONTEXT ---")
    for rank, paper in enumerate(top_papers, 1):
        print(f"Rank {rank} | Cross-Encoder Score: {paper['rerank_score']:.4f}")
        print(f"Title: {paper['title']}")
        print(f"Categories: {paper['categories']} | URL: {paper['pdf_url']}\n" + "-"*50)