import os
import time
import requests
import pandas as pd
from tqdm import tqdm

class ArxivPDFDownloader:
    """
    Downloads raw PDF files from arXiv using the metadata stored in parquet.
    Includes rate-limiting, custom User-Agent, and resumable execution.
    """
    def __init__(self, parquet_path="data/processed/arxiv_dataset.parquet", output_dir="data/raw/pdfs"):
        print("Initializing arXiv PDF Downloader...")
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Cannot find dataset at: {parquet_path}. Please verify the file path.")
            
        self.df = pd.read_parquet(parquet_path)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # arXiv requires a polite User-Agent header to avoid automated blocks
        self.headers = {
            "User-Agent": "ResearchGPT-PDFDownloader/1.0 (Academic Research Pipeline; mailto:research@localhost)"
        }
        print(f"Loaded {len(self.df)} records. Target directory: {self.output_dir}\n")

    def _sanitize_filename(self, paper_id: str) -> str:
        """Replaces problematic characters (like slashes in older arXiv IDs) for local filesystems."""
        return paper_id.replace("/", "_").replace("\\", "_") + ".pdf"

    def download_all(self, delay_seconds: float = 3.0):
        """
        Iterates over the dataset and downloads each PDF.
        Respects arXiv's 3-second delay guideline and skips existing files.
        """
        print(f"Starting download loop with a {delay_seconds}s safety delay between requests...")
        print("Tip: If you stop the script (Ctrl+C), restarting it will automatically resume from where it left off.\n")
        
        success_count = 0
        skipped_count = 0
        failed_ids = []

        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Downloading Papers"):
            paper_id = str(row["id"])
            pdf_url = str(row["pdf_url"])
            
            # Ensure the URL is formatted correctly for direct PDF download
            if not pdf_url.endswith(".pdf") and "arxiv.org/pdf/" in pdf_url:
                pdf_url = pdf_url + ".pdf"

            filename = self._sanitize_filename(paper_id)
            filepath = os.path.join(self.output_dir, filename)

            # 1. Resumability check: skip if file already exists and is larger than 1 KB
            if os.path.exists(filepath) and os.path.getsize(filepath) > 1024:
                skipped_count += 1
                continue

            # 2. Download the file with error handling and stream chunks to disk
            try:
                response = requests.get(pdf_url, headers=self.headers, timeout=30, stream=True)
                
                if response.status_code == 200:
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    success_count += 1
                    
                    # Mandatory safety sleep to respect arXiv server limits
                    time.sleep(delay_seconds)
                else:
                    print(f"\n[Warning] Failed to fetch ID {paper_id} (HTTP {response.status_code})")
                    failed_ids.append(paper_id)
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"\n[Error] Network error on ID {paper_id}: {str(e)}")
                failed_ids.append(paper_id)
                time.sleep(delay_seconds)

        print("\n" + "="*65)
        print(" PDF DOWNLOAD PIPELINE SUMMARY")
        print("="*65)
        print(f"Newly Downloaded: {success_count} papers")
        print(f"Already on Disk (Skipped): {skipped_count} papers")
        print(f"Failed Downloads: {len(failed_ids)} papers")
        print(f"Total Files in '{self.output_dir}': {len(os.listdir(self.output_dir))}")
        
        if failed_ids:
            print(f"\nFailed IDs to retry later: {failed_ids[:10]}...")
        print("="*65)

if __name__ == "__main__":
    downloader = ArxivPDFDownloader()
    downloader.download_all(delay_seconds=3.0)