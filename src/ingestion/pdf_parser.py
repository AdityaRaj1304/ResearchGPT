import os
import re
import fitz # PyMuPDF
import pandas as pd
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page in doc:
        rect = page.rect
        mid_x = rect.width / 2
        
        # Define margins to strip headers/footers/page numbers
        # Usually academic papers have top/bottom margins. We'll strip anything in the top 50 and bottom 50 points.
        top_margin = 50
        bottom_margin = rect.height - 50
        
        # Extract blocks sorted naturally
        blocks = page.get_text("blocks", sort=True)
        
        left_column = []
        right_column = []
        
        for block in blocks:
            x0, y0, x1, y1, text, block_no, block_type = block
            
            # Skip image blocks (block_type == 1)
            if block_type != 0:
                continue
                
            # Skip headers / footers (noise removal)
            if y0 < top_margin or y1 > bottom_margin:
                continue
                
            # Split blocks by page midpoint X-coordinate
            # We use the horizontal center of the block (x0 + x1) / 2
            block_mid_x = (x0 + x1) / 2
            if block_mid_x < mid_x:
                left_column.append(block)
            else:
                right_column.append(block)
                    
        # Sort each column vertically top-to-bottom (y0 coordinate)
        left_column.sort(key=lambda b: b[1])
        right_column.sort(key=lambda b: b[1])
        
        # Append text sequentially
        for b in left_column:
            full_text += b[4].strip() + "\n"
        for b in right_column:
            full_text += b[4].strip() + "\n"
            
    doc.close()
    return full_text

def truncate_bibliography(text):
    total_chars = len(text)
    if total_chars == 0:
        return text, False
        
    threshold = int(total_chars * 0.70)
    
    # We only look at the last 30% of the text
    tail_text = text[threshold:]
    
    # Regex for standalone block headers like "References" or "Bibliography"
    regex = re.compile(r"^\s*10?\s*\.?\s*(References|Bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
    
    match = regex.search(tail_text)
    if not match:
        # Fallback to general number + References block header
        regex = re.compile(r"^\s*\d*\s*\.?\s*(References|Bibliography)\s*$", re.IGNORECASE | re.MULTILINE)
        match = regex.search(tail_text)

    if match:
        truncate_idx = threshold + match.start()
        return text[:truncate_idx].strip(), True
    return text, False

def parse_all_pdfs(input_dir="data/raw/pdfs/", output_path="data/processed/sanitized_texts.parquet", limit=20):
    if not os.path.exists(input_dir):
        print(f"No PDF directory found at {input_dir}")
        return
        
    pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    
    # Enforce Phase A validation limit
    if limit is not None:
        pdf_files = pdf_files[:limit]
        
    print(f"Found {len(pdf_files)} PDFs to parse (Limit set to {limit}).")
    
    if len(pdf_files) == 0:
        print("No PDFs to process.")
        return
        
    results = []
    
    for pdf_file in tqdm(pdf_files, desc="Parsing PDFs"):
        pdf_path = os.path.join(input_dir, pdf_file)
        paper_id = pdf_file.replace(".pdf", "")
        
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count
        doc.close()
        
        raw_text = extract_text_from_pdf(pdf_path)
        sanitized_text, bib_truncated = truncate_bibliography(raw_text)
        
        results.append({
            "paper_id": str(paper_id),
            "raw_pdf_path": str(pdf_path),
            "sanitized_text": str(sanitized_text),
            "total_pages": int(total_pages),
            "char_count": len(sanitized_text),
            "bib_truncated": bool(bib_truncated)
        })
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy")
    print(f"Successfully processed and saved {len(df)} sanitized texts to {output_path}")

if __name__ == "__main__":
    parse_all_pdfs()
