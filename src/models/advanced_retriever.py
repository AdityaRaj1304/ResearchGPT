try:
    import msvc_runtime
except ImportError:
    pass

import os
import time
import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

class ParameterizedRetriever:
    def __init__(self, collection_name: str, reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.collection_name = collection_name
        self.reranker_model_name = reranker_model_name
        self._reranker = None
        
        db_path = "data/vector_store"
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"ChromaDB not found at {db_path}")
            
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name=collection_name)
        
        # We manually load BAAI/bge-small-en-v1.5 to embed the queries 
        # (matching what was used during the build process).
        self.embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        
        # In-memory BM25 index
        print(f"[{collection_name}] Extracting documents and building BM25 index...")
        all_data = self.collection.get(include=["documents", "metadatas"])
        self.documents = all_data["documents"]
        self.metadatas = all_data["metadatas"]
        self.ids = all_data["ids"]
        
        tokenized_corpus = [doc.lower().split() for doc in self.documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        self.doc_map = {
            doc_id: {"doc": doc, "meta": meta, "idx": i} 
            for i, (doc_id, doc, meta) in enumerate(zip(self.ids, self.documents, self.metadatas))
        }

    @property
    def reranker(self):
        # Lazily load to preserve system memory if reranking isn't invoked
        if self._reranker is None:
            print(f"Lazy loading Cross-Encoder ({self.reranker_model_name})...")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def search_dense(self, query: str, top_k: int):
        t_start = time.perf_counter_ns()
        
        # Vectorize query
        t0 = time.perf_counter_ns()
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        query_embedding_ms = (time.perf_counter_ns() - t0) / 1e6
        
        # Dense query
        t0 = time.perf_counter_ns()
        dense_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["distances"]
        )
        dense_search_ms = (time.perf_counter_ns() - t0) / 1e6
        
        candidates = []
        for d_id, dist in zip(dense_results["ids"][0], dense_results["distances"][0]):
            score = 1.0 - dist
            candidates.append({
                "id": d_id,
                "document": self.doc_map[d_id]["doc"],
                "metadata": self.doc_map[d_id]["meta"],
                "dense_score": score,
                "sparse_score": None,
                "rrf_score": None,
                "reranker_score": None,
                "final_score": score
            })
            
        total_retrieval_ms = (time.perf_counter_ns() - t_start) / 1e6
        
        timings = {
            "query_embedding_ms": query_embedding_ms,
            "dense_search_ms": dense_search_ms,
            "sparse_search_ms": 0.0,
            "rrf_merge_ms": 0.0,
            "rerank_ms": 0.0,
            "total_retrieval_ms": total_retrieval_ms
        }
        
        return candidates, timings

    def search_hybrid(self, query: str, top_k: int, candidate_pool_size: int = 20, rrf_k: int = 60):
        t_start = time.perf_counter_ns()
        
        # 1. Embed query
        t0 = time.perf_counter_ns()
        query_emb = self.embedding_model.encode([query], normalize_embeddings=True)[0].tolist()
        query_embedding_ms = (time.perf_counter_ns() - t0) / 1e6
        
        # 2. Dense Search
        t0 = time.perf_counter_ns()
        dense_results = self.collection.query(
            query_embeddings=[query_emb],
            n_results=candidate_pool_size,
            include=["distances"]
        )
        dense_search_ms = (time.perf_counter_ns() - t0) / 1e6
        
        dense_ids = dense_results["ids"][0]
        dense_dists = dense_results["distances"][0]
        dense_scores = {d_id: 1.0 - dist for d_id, dist in zip(dense_ids, dense_dists)}
        
        # 3. Sparse Search
        t0 = time.perf_counter_ns()
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_sparse_indices = np.argsort(bm25_scores)[::-1][:candidate_pool_size]
        sparse_search_ms = (time.perf_counter_ns() - t0) / 1e6
        
        sparse_scores = {self.ids[idx]: bm25_scores[idx] for idx in top_sparse_indices}
        
        # 4. RRF Merge
        t0 = time.perf_counter_ns()
        combined_ids = set(list(dense_scores.keys()) + list(sparse_scores.keys()))
        
        dense_rank = {d_id: rank + 1 for rank, d_id in enumerate(dense_ids)}
        sparse_ids_sorted = [self.ids[idx] for idx in top_sparse_indices]
        sparse_rank = {d_id: rank + 1 for rank, d_id in enumerate(sparse_ids_sorted)}
        
        candidates = []
        for d_id in combined_ids:
            r_dense = dense_rank.get(d_id, None)
            r_sparse = sparse_rank.get(d_id, None)
            
            val_dense = 1.0 / (rrf_k + r_dense) if r_dense is not None else 0.0
            val_sparse = 1.0 / (rrf_k + r_sparse) if r_sparse is not None else 0.0
            rrf_score = val_dense + val_sparse
            
            candidates.append({
                "id": d_id,
                "document": self.doc_map[d_id]["doc"],
                "metadata": self.doc_map[d_id]["meta"],
                "dense_score": dense_scores.get(d_id, None),
                "sparse_score": sparse_scores.get(d_id, None),
                "rrf_score": rrf_score,
                "reranker_score": None,
                "final_score": rrf_score
            })
            
        candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
        rrf_merge_ms = (time.perf_counter_ns() - t0) / 1e6
        
        total_retrieval_ms = (time.perf_counter_ns() - t_start) / 1e6
        
        timings = {
            "query_embedding_ms": query_embedding_ms,
            "dense_search_ms": dense_search_ms,
            "sparse_search_ms": sparse_search_ms,
            "rrf_merge_ms": rrf_merge_ms,
            "rerank_ms": 0.0,
            "total_retrieval_ms": total_retrieval_ms
        }
        
        return candidates[:top_k], timings

    def search_hybrid_rerank(self, query: str, top_k: int, candidate_pool_size: int = 20, rrf_k: int = 60):
        t_start = time.perf_counter_ns()
        
        # Get preliminary pool of RRF candidates (up to 2*candidate_pool_size)
        candidates, timings = self.search_hybrid(
            query=query, 
            top_k=candidate_pool_size * 2, 
            candidate_pool_size=candidate_pool_size, 
            rrf_k=rrf_k
        )
        
        # Rerank with Cross-Encoder
        t0 = time.perf_counter_ns()
        model_inputs = [[query, c["document"]] for c in candidates]
        rerank_scores = self.reranker.predict(model_inputs)
        rerank_ms = (time.perf_counter_ns() - t0) / 1e6
        
        for i, c in enumerate(candidates):
            c["reranker_score"] = float(rerank_scores[i])
            c["final_score"] = float(rerank_scores[i])
            
        candidates.sort(key=lambda x: x["final_score"], reverse=True)
        
        total_retrieval_ms = (time.perf_counter_ns() - t_start) / 1e6
        
        # Update timings correctly (excluding the internal hybrid timing wrapper total)
        timings["rerank_ms"] = rerank_ms
        timings["total_retrieval_ms"] = total_retrieval_ms
        
        return candidates[:top_k], timings