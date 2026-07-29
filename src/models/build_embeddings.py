import msvc_runtime
import os
import time
import json
import argparse
import datetime
import pandas as pd
import chromadb
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DATASET_COLLECTION_MAP = {
    "whole_doc": {"file": "chunks_whole_doc.parquet", "collection": "arxiv_wholedoc"},
    "fixed": {"file": "chunks_fixed.parquet", "collection": "arxiv_fixed_512"},
    "overlap": {"file": "chunks_overlap.parquet", "collection": "arxiv_overlap_512_128"},
    "semantic": {"file": "chunks_semantic.parquet", "collection": "arxiv_semantic"}
}

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        return "xpu"
    else:
        return "cpu"

def build_vector_database(batch_size=64):
    device = get_device()
    model_name = "BAAI/bge-small-en-v1.5"
    print(f"Embedding Model : {model_name}")
    print(f"Device          : {device} (auto-detected)")
    print(f"Batch Size      : {batch_size}")

    db_path = "data/vector_store"
    os.makedirs(db_path, exist_ok=True)
    
    arxiv_meta_path = "data/processed/arxiv_dataset.parquet"
    if os.path.exists(arxiv_meta_path):
        meta_df = pd.read_parquet(arxiv_meta_path)
        meta_dict = pd.Series(meta_df['title'].values, index=meta_df['id']).to_dict()
    else:
        meta_dict = {}

    client = chromadb.PersistentClient(path=db_path)
    encoder = SentenceTransformer(model_name, device=device)

    for strategy, config in DATASET_COLLECTION_MAP.items():
        input_parquet = os.path.join("data/processed", config["file"])
        collection_name = config["collection"]
        
        print(f"\nProcessing strategy: {strategy}")
        if not os.path.exists(input_parquet):
            print(f"Skipping {strategy}: {input_parquet} not found.")
            continue
            
        df = pd.read_parquet(input_parquet)
        total_records = len(df)
        print(f"Loaded {total_records} records for {strategy}.")
        
        existing_collections = [c.name for c in client.list_collections()]
        if collection_name in existing_collections:
            client.delete_collection(collection_name)
            
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        total_embed_time = 0.0
        total_insert_time = 0.0
        
        for start_idx in tqdm(range(0, total_records, batch_size), desc=f"Indexing {collection_name}"):
            end_idx = min(start_idx + batch_size, total_records)
            batch_df = df.iloc[start_idx:end_idx]
            
            batch_texts = batch_df["text"].tolist()
            
            batch_ids = []
            batch_metadata = []
            
            for _, row in batch_df.iterrows():
                paper_id = str(row["paper_id"])
                chunk_id_raw = str(row["chunk_id"])
                
                if "_" in chunk_id_raw:
                    try:
                        chunk_index = int(chunk_id_raw.rsplit("_", 1)[-1])
                    except ValueError:
                        chunk_index = 0
                else:
                    chunk_index = 0
                    
                vector_id = f"{strategy}_{paper_id}_c{chunk_index:04d}"
                batch_ids.append(vector_id)
                
                meta = {
                    "paper_id": paper_id,
                    "paper_title": meta_dict.get(paper_id, "Unknown Title"),
                    "chunk_id": vector_id,
                    "chunk_index": chunk_index,
                    "chunk_strategy": strategy,
                    "token_count": int(row["token_length"]),
                    "section_name": "Unknown",
                    "page_start": -1,
                    "page_end": -1,
                    "collection_name": collection_name
                }
                batch_metadata.append(meta)
                
            t0 = time.time()
            batch_embeddings = encoder.encode(batch_texts, show_progress_bar=False, normalize_embeddings=True)
            batch_embeddings = batch_embeddings.tolist()
            total_embed_time += (time.time() - t0)
            
            t1 = time.time()
            collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
                documents=batch_texts
            )
            total_insert_time += (time.time() - t1)
            
        print(f"Total rows currently indexed: {collection.count()}")
        
        manifest = {
            "collection_name": collection_name,
            "chunk_strategy": strategy,
            "embedding_model": model_name,
            "embedding_dimension": encoder.get_sentence_embedding_dimension(),
            "distance_metric": "cosine",
            "total_vectors": collection.count(),
            "device_used": device,
            "batch_size": batch_size,
            "build_timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "total_embed_time_s": round(total_embed_time, 2),
            "total_insert_time_s": round(total_insert_time, 2),
            "total_build_time_s": round(total_embed_time + total_insert_time, 2)
        }
        
        manifest_path = os.path.join(db_path, f"{collection_name}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
            
        print(f"Manifest written to: {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for embedding generation")
    args = parser.parse_args()
    
    start_timer = time.time()
    build_vector_database(batch_size=args.batch_size)
    print(f"Total pipeline execution time: {round(time.time() - start_timer, 2)} seconds.")