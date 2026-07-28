import msvc_runtime
import os
import pandas as pd
from tqdm import tqdm
import time
import numpy as np
from src.ingestion.chunkers import WholeDocChunker, FixedSizeChunker, OverlapChunker, SemanticChunker
from transformers import AutoTokenizer

def profile_chunkers(input_parquet="data/processed/sanitized_texts.parquet"):
    print(f"Loading {input_parquet}...")
    df = pd.read_parquet(input_parquet)
    
    print("Initializing chunkers...")
    chunkers = {
        "whole_doc": WholeDocChunker(),
        "fixed": FixedSizeChunker(),
        "overlap": OverlapChunker(),
        "semantic": SemanticChunker()
    }
    
    # We will compute lengths based on token count using the common tokenizer
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    
    stats_list = []
    
    for name, chunker in chunkers.items():
        print(f"\nRunning {name} chunking strategy...")
        start_time = time.time()
        
        # Prepare output dataframe
        out_rows = []
        chunk_lengths = []
        chunks_per_paper = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            paper_id = row['paper_id']
            text = row['sanitized_text']
            
            chunks = chunker.chunk(text)
            chunks_per_paper.append(len(chunks))
            
            for chunk_idx, chunk_text in enumerate(chunks):
                token_len = len(tokenizer.encode(chunk_text, add_special_tokens=False))
                chunk_lengths.append(token_len)
                
                out_rows.append({
                    "paper_id": paper_id,
                    "chunk_id": f"{paper_id}_{chunk_idx}",
                    "text": chunk_text,
                    "token_length": token_len
                })
                
        # Save output parquet
        out_df = pd.DataFrame(out_rows)
        output_path = f"data/processed/chunks_{name}.parquet"
        out_df.to_parquet(output_path)
        print(f"Saved to {output_path}")
        
        # Calculate stats
        stats_list.append({
            "Strategy": name,
            "Total Chunks": len(chunk_lengths),
            "Avg Chunks/Paper": np.mean(chunks_per_paper),
            "Avg Token Length": np.mean(chunk_lengths),
            "Max Token Length": np.max(chunk_lengths),
            "Min Token Length": np.min(chunk_lengths),
            "Std Token Length": np.std(chunk_lengths),
            "Time (s)": round(time.time() - start_time, 2)
        })
        
    stats_df = pd.DataFrame(stats_list)
    stats_csv_path = "data/processed/chunk_statistics.csv"
    stats_df.to_csv(stats_csv_path, index=False)
    
    print("\nChunking Profiling Complete. Stats:")
    print(stats_df)

if __name__ == "__main__":
    profile_chunkers()
