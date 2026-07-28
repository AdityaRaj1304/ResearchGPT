import os
import time
import urllib.request
import urllib.error
import pandas as pd
from tqdm import tqdm

def fetch_pdfs(parquet_path="data/processed/arxiv_dataset.parquet", output_dir="data/raw/pdfs/", limit=20):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading metadata from {parquet_path}...")
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"{parquet_path} not found. Please ensure the metadata is downloaded.")
        
    df = pd.read_parquet(parquet_path)
    
    papers_to_fetch = df.head(limit)
    print(f"Fetching {len(papers_to_fetch)} PDFs...")
    
    headers = {
        'User-Agent': 'ResearchGPT-AblationStudy/1.0 (Contact: research@example.com)'
    }
    
    for _, row in tqdm(papers_to_fetch.iterrows(), total=len(papers_to_fetch)):
        paper_id = row['id']
        # Arxiv IDs might have slashes if old, replace them to safely save files
        safe_id = str(paper_id).replace("/", "_")
        pdf_url = row['pdf_url']
        
        # Sometime arxiv API returns http instead of https, let's enforce https
        if pdf_url.startswith("http://"):
            pdf_url = pdf_url.replace("http://", "https://")
        
        # Also ensure pdf extension is present, pdf_url is usually https://arxiv.org/pdf/1234.5678v1
        pdf_url += ".pdf"
        
        output_path = os.path.join(output_dir, f"{safe_id}.pdf")
        
        if os.path.exists(output_path):
            continue
            
        # Retry loop for 429
        max_retries = 3
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(pdf_url, headers=headers)
                with urllib.request.urlopen(req) as response, open(output_path, 'wb') as out_file:
                    out_file.write(response.read())
                break # Success
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    print(f"\n[HTTP 429] Too Many Requests for {paper_id}. Retrying in 10s (Attempt {attempt+1}/{max_retries})...")
                    time.sleep(10)
                else:
                    print(f"\nHTTP Error {e.code} for {paper_id}")
                    break
            except Exception as e:
                print(f"\nError fetching {paper_id}: {e}")
                break
                
        # Mandatory 3s delay
        time.sleep(3)

if __name__ == "__main__":
    fetch_pdfs()
