import os
import arxiv
import pandas as pd
from tqdm import tqdm

def fetch_sample_papers(query="cat:cs.CL OR cat:cs.LG", max_results=5):
    print(f"🔍 Searching arXiv for: '{query}'...")
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate
    )
    
    papers_data = []
    for result in tqdm(client.results(search), total=max_results, desc="Fetching Metadata"):
        papers_data.append({
            "paper_id": result.get_short_id().split('v')[0],
            "title": result.title,
            "authors": ", ".join([a.name for a in result.authors]),
            "published": result.published.strftime("%Y-%m-%d"),
            "summary": result.summary.replace("\n", " ")
        })
        
    df = pd.DataFrame(papers_data)
    
    # Save a quick sample to Parquet to test PyArrow
    output_path = "data/processed/sample_papers.parquet"
    df.to_parquet(output_path, engine="pyarrow")
    
    print(f"\n✅ Successfully fetched {len(df)} papers!")
    print(f"📁 Saved test dataset to: {output_path}")
    print("\n--- Preview ---")
    print(df[["paper_id", "title", "published"]].to_string(index=False))

if __name__ == "__main__":
    fetch_sample_papers()