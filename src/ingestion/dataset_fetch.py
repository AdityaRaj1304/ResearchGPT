import os
import time
import arxiv
import pandas as pd
from tqdm import tqdm

def bulk_fetch_arxiv_data(total_results=1000, batch_size=100, output_path="data/processed/arxiv_dataset.parquet"):
    """
    Fetches papers from arXiv in batches with rate-limiting and saves them to a Parquet file.
    """
    # 1. Initialize the arXiv Client
    client = arxiv.Client()
    
    # Define search query focusing on NLP/LLMs (cs.CL) and Machine Learning (cs.LG)
    # Sorted by newest submissions first
    search = arxiv.Search(
        query="cat:cs.CL OR cat:cs.LG",
        max_results=total_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    all_papers = []
    
    # 2. Ensure destination directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Starting bulk download of {total_results} papers from arXiv...")
    print(f"Batch Size: {batch_size} |  Rate Limit Pause: 3s\n")
    
    # 3. Create a progress bar across the total number of papers
    with tqdm(total=total_results, desc="Downloading Papers") as pbar:
        # Use generator to fetch results lazily or handle chunking manually
        results_generator = client.results(search)
        
        batch_counter = 0
        current_batch = []
        
        try:
            for paper in results_generator:
                # Extract the required schema fields for your database
                paper_data = {
                    "id": paper.get_short_id(),
                    "title": paper.title,
                    "abstract": paper.summary.replace("\n", " "), # Clean up newline tokens
                    "authors": [author.name for author in paper.authors],
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "categories": paper.categories,
                    "pdf_url": paper.pdf_url
                }
                
                current_batch.append(paper_data)
                pbar.update(1)
                
                # When a batch fills up, pause to respect rate limits
                if len(current_batch) == batch_size:
                    all_papers.extend(current_batch)
                    current_batch = [] # Reset batch
                    batch_counter += 1
                    
                    # Prevent a sleep on the final batch completion
                    if len(all_papers) < total_results:
                        # Polite API etiquette pause
                        time.sleep(3)
                        
            # Append any remaining papers if total results wasn't perfectly divisible by batch_size
            if current_batch:
                all_papers.extend(current_batch)
                
        except Exception as e:
            print(f"\n An error occurred during transmission: {e}")
            print("Saving partial dataset collected so far to prevent loss...")
            
    # 4. Save to Parquet format if we have data
    if all_papers:
        df = pd.DataFrame(all_papers)
        
        # Deduplicate results just in case
        df.drop_duplicates(subset=["id"], inplace=True)
        
        df.to_parquet(output_path, engine="pyarrow", compression="snappy")
        print(f"\n Successfully harvested {len(df)} unique papers!")
        print(f" Dataset written to disk at: {output_path}")
    else:
        print("\nNo papers collected.")

if __name__ == "__main__":
    # Configure parameters
    TOTAL_PAPERS = 1000
    BATCH_SIZE = 100
    OUTPUT_FILE = "data/processed/arxiv_dataset.parquet"
    
    bulk_fetch_arxiv_data(
        total_results=TOTAL_PAPERS, 
        batch_size=BATCH_SIZE, 
        output_path=OUTPUT_FILE
    )