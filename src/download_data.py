# src/download_data.py
import os
import arxiv
from tqdm import tqdm

def download_initial_dataset(query="cat:cs.CL OR cat:cs.LG", max_results=50):
    os.makedirs("data/raw", exist_ok=True)
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    print(f"Starting download of {max_results} papers for query: '{query}'...")
    for result in tqdm(client.results(search), total=max_results, desc="Downloading"):
        clean_id = result.get_short_id().split('v')[0]
        pdf_path = f"data/raw/{clean_id}.pdf"
        
        if not os.path.exists(pdf_path):
            result.download_pdf(dirpath="data/raw", filename=f"{clean_id}.pdf")

if __name__ == "__main__":
    # Start small with 50 papers today to verify everything works perfectly
    download_initial_dataset(max_results=50)