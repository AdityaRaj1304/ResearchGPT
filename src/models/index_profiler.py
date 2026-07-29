import msvc_runtime
import os
import json
import pandas as pd
import chromadb

DATASET_COLLECTION_MAP = {
    "whole_doc": {"file": "chunks_whole_doc.parquet", "collection": "arxiv_wholedoc"},
    "fixed": {"file": "chunks_fixed.parquet", "collection": "arxiv_fixed_512"},
    "overlap": {"file": "chunks_overlap.parquet", "collection": "arxiv_overlap_512_128"},
    "semantic": {"file": "chunks_semantic.parquet", "collection": "arxiv_semantic"}
}

def get_dir_size(path):
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def profile_indices(db_path="data/vector_store"):
    if not os.path.exists(db_path):
        print(f"DB path {db_path} does not exist.")
        return
        
    client = chromadb.PersistentClient(path=db_path)
    existing_collections = [c.name for c in client.list_collections()]
    
    results = []
    
    total_db_size_mb = get_dir_size(db_path) / (1024 * 1024)
    total_vectors_in_db = sum(client.get_collection(c).count() for c in existing_collections)
    
    print("\nStarting Index Profiling...\n")
    
    for strategy, config in DATASET_COLLECTION_MAP.items():
        input_parquet = os.path.join("data/processed", config["file"])
        collection_name = config["collection"]
        
        if collection_name not in existing_collections:
            print(f"Collection {collection_name} not found in DB.")
            continue
            
        print(f"Profiling {collection_name}...")
        
        df = pd.read_parquet(input_parquet)
        expected_count = len(df)
        
        collection = client.get_collection(collection_name)
        actual_count = collection.count()
        assert expected_count == actual_count, f"[{collection_name}] Vector Count ({actual_count}) != Parquet Chunk Count ({expected_count})"
        
        all_data = collection.get(include=["embeddings", "metadatas", "documents"])
        
        embeddings = all_data.get("embeddings")
        metadatas = all_data.get("metadatas")
        ids = all_data.get("ids")
        
        assert embeddings is not None and len(embeddings) > 0, f"[{collection_name}] Zero empty embeddings check failed."
        assert len(embeddings) == expected_count, f"[{collection_name}] Embeddings length mismatch"
        
        for emb in embeddings:
            assert len(emb) == 384, f"[{collection_name}] Invalid vector dimension: {len(emb)}"
            
        required_keys = ["paper_id", "paper_title", "chunk_id", "chunk_index", "chunk_strategy", "token_count", "section_name", "page_start", "page_end", "collection_name"]
        for meta in metadatas:
            for key in required_keys:
                assert key in meta, f"[{collection_name}] Metadata key '{key}' missing."
                
        assert len(set(ids)) == len(ids), f"[{collection_name}] Duplicate primary keys found."
        
        manifest_path = os.path.join(db_path, f"{collection_name}_manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        embed_time_s = manifest.get("total_embed_time_s", 0)
        insert_time_s = manifest.get("total_insert_time_s", 0)
        build_time_s = manifest.get("total_build_time_s", 0)
        
        avg_embed_ms = (embed_time_s / actual_count) * 1000 if actual_count else 0
        avg_insert_ms = (insert_time_s / actual_count) * 1000 if actual_count else 0
        
        storage_footprint_mb = total_db_size_mb * (actual_count / total_vectors_in_db) if total_vectors_in_db > 0 else 0
        storage_efficiency = storage_footprint_mb / (actual_count / 1000) if actual_count > 0 else 0
        
        results.append({
            "Collection": collection_name,
            "Vectors": actual_count,
            "Total Storage Footprint (MB)": round(storage_footprint_mb, 2),
            "Storage Efficiency (MB per 1k)": round(storage_efficiency, 2),
            "Total Build Time (s)": round(build_time_s, 2),
            "Avg Embedding Time (ms/vec)": round(avg_embed_ms, 2),
            "Avg Insertion Time (ms/vec)": round(avg_insert_ms, 2)
        })
        
    print("\nData integrity validation passed for all collections (Zero errors).")
    
    stats_df = pd.DataFrame(results)
    
    out_path = "data/processed/index_statistics.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    stats_df.to_csv(out_path, index=False)
    
    print("\nSummary Table:")
    print(stats_df.to_markdown(index=False))
    print(f"\nSaved metrics to {out_path}")

if __name__ == "__main__":
    profile_indices()
