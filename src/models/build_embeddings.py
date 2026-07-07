import msvc_runtime
import os
import time
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm

def build_vector_database(
    input_parquet="data/processed/arxiv_dataset.parquet",
    db_path="data/vector_store",
    collection_name="arxiv_papers",
    batch_size=64
):
    """
    Loads parquet data, structures content strings, generates ONNX embeddings,
    and indexes them into a local ChromaDB instance without PyTorch dependencies.
    """
    
    print(f"Loading dataset from: {input_parquet}")
    if not os.path.exists(input_parquet):
        raise FileNotFoundError(f"Cannot find dataset at {input_parquet}. Please run your download script first.")
        
    df = pd.read_parquet(input_parquet)
    print(f"Successfully loaded {len(df)} records from disk.")
    
    # 1. Structure the embedding input with explicit markers
    print("Constructing structured context text column (Title + Abstract)...")
    df["text"] = "Title: " + df["title"] + " | Abstract: " + df["abstract"]
    
    # 2. Initialize the PyTorch embedding engine (all-MiniLM-L6-v2)
    print("Initializing PyTorch embedding engine (all-MiniLM-L6-v2)...")
    onnx_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    # 3. Initialize ChromaDB persistent storage layer
    print(f"Setting up persistent ChromaDB at: {db_path}...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Delete the collection if it exists to allow for a clean overwrite/re-index
    existing_collections = [c.name for c in client.list_collections()]
    if collection_name in existing_collections:
        print(f"Resetting existing collection: {collection_name}")
        client.delete_collection(collection_name)
        
    collection = client.create_collection(
        name=collection_name,
        embedding_function=onnx_ef,
        metadata={"hnsw:space": "cosine"} # Explicitly configure Cosine Similarity
    )
    
    # 4. Generate embeddings and index in batches
    print(f"Beginning text vectorization via ONNX (Batch Size: {batch_size})...")
    total_records = len(df)
    
    for start_idx in tqdm(range(0, total_records, batch_size), desc="Processing Batches"):
        end_idx = min(start_idx + batch_size, total_records)
        batch_df = df.iloc[start_idx:end_idx]
        
        batch_ids = batch_df["id"].tolist()
        batch_texts = batch_df["text"].tolist()
        
        # Metadata must use simple primitive types (strings, numbers)
        batch_metadata = [
            {
                "title": row["title"],
                "published": str(row["published"]),
                "categories": ", ".join(row["categories"]) if isinstance(row["categories"], list) else str(row["categories"]),
                "pdf_url": row["pdf_url"]
            }
            for _, row in batch_df.iterrows()
        ]
        
        # Compute embeddings for the batch
        batch_embeddings = onnx_ef(batch_texts)
        
        # Write into ChromaDB
        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            metadatas=batch_metadata,
            documents=batch_texts
        )
        
    print("\nVector indexing operations completed successfully.")
    print(f"Total rows currently indexed: {collection.count()}")
    print(f"Database checkpoint saved locally at: {db_path}")

if __name__ == "__main__":
    INPUT_FILE = "data/processed/arxiv_dataset.parquet"
    DB_DIRECTORY = "data/vector_store"
    
    start_timer = time.time()
    build_vector_database(input_parquet=INPUT_FILE, db_path=DB_DIRECTORY)
    execution_duration = round(time.time() - start_timer, 2)
    print(f"Total pipeline execution time: {execution_duration} seconds.")