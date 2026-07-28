# **Antigravity Task Specification: ResearchGPT Granular Ablation Study** 

**AGENT INSTRUCTION NOTICE:** This document is an architectural specification and implementation roadmap. Your current task is strictly **READ-ONLY** . Do NOT generate code, modify existing files, or execute terminal commands. Review this specification, analyze the current project structure against these requirements, and await explicit user authorization before initiating Phase 1. 

## **1. Project Objective & Core Research Questions** 

The objective of this project is to execute a rigorous experimental study evaluating how document segmentation strategies, retrieval architectures, and context sizes ( _Top_ - _K_ ) impact retrieval quality, latency, generation faithfulness, and citation integrity in academic RAG systems processing full-length research PDFs. 

#### **Primary Research Questions:** 

- **RQ1:** How significantly do chunking strategies (Whole-Document baseline vs. Fixed vs. Overlap vs. Formal Semantic) improve retrieval quality (Precision@K, Recall@K, MRR, nDCG) compared to un-chunked full-paper representations? 

- **RQ2:** What is the marginal retrieval benefit of applying a neural Cross-Encoder reranker across distinct chunk distributions and context cutoff sizes ( _K_ =3 vs. _K_ =5)? 

- **RQ3:** How do different chunking algorithms impact generation quality, specifically citation accuracy and hallucination rate? 

- **RQ4:** Are observed performance variations between chunking and retrieval configurations statistically significant ( _p_ <0.05)? 

## **2. Architectural Scope & Scaled Dataset Pipeline** 

The codebase will be extended to support a phased dataset scaling model: 

- **Phase A (Validation Set):** 20 full-text PDFs (for rapid code verification and integration testing). 

- **Phase B (Experimental Scale):** 200 full-text PDFs (for main ablation runs and hyperparameter validation). 

- **Phase C (Final Benchmark Scale):** 1,000 full-text PDFs (for final statistical significance reporting and paper figures). 

## **3. Core Component Specifications** 

### **A. Full-PDF Extraction & Layout Sanitization** 

- **Extraction Engine:** Use PyMuPDF (fitz) or pdfplumber to process multi-column academic paper structures. 

- **Sanitization Requirements:** 

   - Strip headers, footers, page numbering, and margin artifacts. 

   - Detect and isolate or strip raw bibliography/reference lists to eliminate keyword retrieval noise. 

   - Standardize reading order across two-column layouts. 

### **B. Segmenter Engine (4 Strategies)** 

1. **Whole-Document Baseline (No Chunking):** Single embedding per entire paper (collapsing full sanitized text into one representation). 

2. **Fixed-Size Chunking:** Hard boundaries at **512 tokens** (using all-MiniLM-L6-v2 tokenizer) with **0 token overlap** . 

3. **Overlapping Chunking: 512-token chunks** with a **128-token sliding window overlap** (384-token step size). 

4. **Formal Semantic Chunking Engine:** 

   - Step 1: Detect explicit section headers ( _Abstract_ , _Introduction_ , _Methodology_ , _Experiments_ , _Results_ , _Conclusion_ ). 

   - Step 2: Perform sentence segmentation using PyTorch/Spacy/NLTK. 

   - Step 3: Compute adjacent sentence vector embeddings using all-MiniLM-L6v2. 

   - Step 4: Calculate rolling cosine similarity across adjacent sentence vectors. 

   - Step 5: Place chunk boundaries at local similarity drops below a dynamic percentile threshold (or hard minimum similarity score). 

### **C. Chunk Profiler Engine** 

For each chunking strategy across the ingestion pipeline, calculate and export structural metadata: 

- Average chunk token length 

- Maximum and minimum chunk token length 

- Standard deviation of chunk lengths 

- Total chunk count per paper 

- Distribution histogram data 

### **D. Parameterized Storage Layer (4 ChromaDB Collections)** 

The indexing engine must write to four dedicated collections within data/vector_store: 

1. arxiv_full_doc_baseline 

2. arxiv_full_fixed_512 

3. arxiv_full_overlap_512_128 

4. arxiv_full_semantic_formal 

Each chunk record must retain: paper_id, paper_title, chunk_index, chunk_strategy, token_count, and section_name. 

## **4. The 24-Pipeline Experimental Matrix (**<sup>4</sup><sup>_×_3</sup><sup>_×_2</sup> **)** 

The system will systematically benchmark 24 pipeline configurations across **4 Chunking Methods** _×_ **3 Retrieval Architectures** _×_ **2 Context Cutoff Sizes (** _Top_ - _K_ **)** : 

- **Chunking Methods:** Whole-Doc, Fixed, Overlap, Semantic 

- **Retrievers:** Dense Only, Hybrid (Dense + BM25), Hybrid + Reranker (ms-marco-MiniLM-L-6v2) 

- **Context Cutoffs:** Top-K = 3, Top-K = 5 

|**Config**<br>**ID**|**Chunking**<br>**Strategy**|**Retrieval**<br>**Strategy**|**Reranker**|**Context Cutoff**<br>**(Top-K)**|
|---|---|---|---|---|
|**C01 -**<br>**C02**|Whole<br>Document|Dense|No|_K_=3_, K_=5|
|**C03 -**<br>**C04**|Whole<br>Document|Hybrid|No|_K_=3_, K_=5|
|**C05 -**<br>**C06**|Whole<br>Document|Hybrid|Yes|_K_=3_, K_=5|
|**C07 -**<br>**C08**|Fixed-Size|Dense|No|_K_=3_, K_=5|
|**C09 -**<br>**C10**|Fixed-Size|Hybrid|No|_K_=3_, K_=5|
|**C11 -**<br>**C12**|Fixed-Size|Hybrid|Yes|_K_=3_, K_=5|
|**C13 -**<br>**C14**|Overlapping|Dense|No|_K_=3_, K_=5|
|**C15 -**<br>**C16**|Overlapping|Hybrid|No|_K_=3_, K_=5|
|**C17 -**<br>**C18**|Overlapping|Hybrid|Yes|_K_=3_, K_=5|
|**C19 -**<br>**C20**|Formal Semantic|Dense|No|_K_=3_, K_=5|



|**Config**<br>**ID**|**Chunking**<br>**Strategy**|**Retrieval**<br>**Strategy**|**Reranker**|**Context Cutoff**<br>**(Top-K)**|
|---|---|---|---|---|
|**C21 -**<br>**C22**|Formal Semantic|Hybrid|No|_K_=3_, K_=5|
|**C23 -**<br>**C24**|Formal Semantic|Hybrid|Yes|_K_=3_, K_=5|



## **5. Comprehensive Evaluation Framework** 

### **A. Retrieval Performance Metrics** 

- **Precision@K (** _K ∈_ {3,5 } **)** 

- **Recall@K (** _K ∈_ {3,5,10 } **)** 

- **Mean Reciprocal Rank (MRR)** 

- **Normalized Discounted Cumulative Gain (nDCG@K)** 

### **B. Granular System Efficiency & Resource Profiling** 

Track and log performance bottlenecks separately: 

- **Index Disk Footprint** (MB per collection) 

- **Embedding Ingestion Time** (seconds per 100 papers) 

- **Search Latency** (milliseconds) 

- **Rerank Latency** (milliseconds) 

- **LLM Generation Latency** (milliseconds) 

- **End-to-End Latency** (milliseconds) 

### **C. Generation & Quality Metrics** 

- **Citation Accuracy Score:** Percentage of claims in the generated response that are directly entailed by the retrieved chunk text cited. 

- **Hallucination Rate:** Percentage of generated claims that lack ground-truth context in the retrieved chunks. 

### **D. Statistical Hypothesis Testing** 

Automate pairwise statistical significance analysis across configurations: 

- **Paired t-test** (for normally distributed metric comparisons) 

- **Wilcoxon Signed-Rank Test** (non-parametric comparison) 

- Report _p_ -values and confidence intervals to verify whether performance delta between Whole Doc vs Semantic and Hybrid vs Hybrid+Reranker is statistically significant ( _p_ <0.05). 

## **6. Multi-Phase Execution Roadmap** 

### **Phase 1: Full PDF Extraction & Sanitization** 

- Upgrade dataset ingest pipeline to fetch full PDF binaries. 

- Implement layout parsing and noise removal scripts. 

- Validate text extraction accuracy on 20-paper dataset before expanding to 200 and 1,000 papers. 

### **Phase 2: Chunking Engines & Profiling Suite** 

- Implement Whole Doc, Fixed, Overlap, and Formal Semantic chunking logic in src/ingestion/chunkers.py. 

- Build chunk distribution statistical profiler (averages, counts, token length variance). 

### **Phase 3: Dynamic Multi-Index Storage** 

- Parameterize vector storage scripts to construct and populate four isolated ChromaDB collections. 

- Log index construction timing and storage size footprints across strategies. 

### **Phase 4: Parameterized Retrieval Engine** 

- Refactor advanced_retriever.py to support dynamic collection switching, BM25 indexing per chunking strategy, and variable _Top_ - _K_ parameters. 

### **Phase 5: Automated Benchmarking & Generation Evaluator** 

- Upgrade evaluate_retrieval.py to execute all 24 experimental runs against benchmark_queries.json. 

- Integrate timer instrumentation for search, rerank, and generation steps. 

- Integrate LLM-as-a-judge verification loop for Citation Accuracy and Hallucination Rate scoring. 

### **Phase 6: Statistical Analysis & Paper Output Generation** 

- Implement src/evaluation/statistical_tests.py for automated paired t-tests and Wilcoxon tests. 

- Export structured comparison tables (CSV, JSON) and generate publicationready performance graphs. 

