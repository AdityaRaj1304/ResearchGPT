# ResearchGPT

> Autonomous Academic Research & Synthesis Engine

ResearchGPT is a production-oriented Retrieval-Augmented Generation (RAG) system designed for academic research. Unlike conventional RAG systems that rely solely on vector similarity, ResearchGPT implements a multi-stage retrieval architecture combining dense retrieval, sparse retrieval, and neural reranking to deliver highly relevant, citation-grounded responses.

---

## Features

### Hybrid Retrieval Pipeline

Combines:

- Dense Semantic Search (ChromaDB + MiniLM Embeddings)
- Sparse Lexical Search (BM25)

This enables retrieval of both:

- Semantically relevant papers
- Exact technical terminology

---

### Neural Reranking

Uses:

- `cross-encoder/ms-marco-MiniLM-L-6-v2`

The reranker performs deep query-document attention and reorders candidate papers retrieved from the hybrid search stage.

Pipeline:

```text
Query
   ↓
Dense Retrieval
   ↓
BM25 Retrieval
   ↓
Hybrid Candidate Set
   ↓
Cross Encoder Reranker
   ↓
Top Relevant Papers
```

### Citation-Grounded Generation

Responses are generated using retrieved academic sources and include paper references to reduce hallucinations.

### Local-First Architecture

Runs completely on local hardware using:

- Ollama
- Llama 3
- ChromaDB

No external API calls required.

---

## Current Architecture

```text
arXiv Papers
      ↓
Parquet Dataset
      ↓
Text Processing
      ↓
MiniLM Embeddings
      ↓
ChromaDB
      ↓
Hybrid Retrieval
      ├── Dense Search
      └── BM25 Search
      ↓
Cross Encoder Reranking
      ↓
Top-K Context
      ↓
Llama 3 (Ollama)
      ↓
Citation-Grounded Answer
```

---

## Dataset

Current Corpus:

- 1000+ Research Papers
- Domains:
  - Machine Learning
  - Natural Language Processing

Metadata Stored:

- Paper ID
- Title
- Abstract
- Authors
- Publication Date
- Categories
- PDF URL

Source:

- arXiv API

---

## Tech Stack

### Retrieval

- ChromaDB
- rank-bm25

### Embeddings

- all-MiniLM-L6-v2

### Reranking

- cross-encoder/ms-marco-MiniLM-L-6-v2

### LLM

- Ollama
- Llama 3

### Backend

- Python 3.12+

---

## Repository Structure

```text
ResearchGPT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
│
├── src/
│   ├── ingestion/
│   ├── retrieval/
│   ├── reranking/
│   ├── generation/
│   └── evaluation/
│
├── notebooks/
│
├── tests/
│
├── requirements.txt
│
└── README.md
```

---

## Roadmap

### Phase 1 — Core RAG

- [x] arXiv ingestion
- [x] Embedding generation
- [x] ChromaDB indexing
- [x] BM25 retrieval
- [x] Hybrid search
- [x] Cross-encoder reranking
- [x] Citation-based answering

### Phase 2 — Evaluation Framework

- [ ] Precision@K
- [ ] Recall@K
- [ ] MRR
- [ ] nDCG
- [ ] Retrieval benchmarking

### Phase 3 — Full Paper Intelligence

- [ ] PDF parsing
- [ ] Section-aware chunking
- [ ] Citation graph
- [ ] Related paper recommendations

### Phase 4 — Research Intelligence

- [ ] Paper comparison
- [ ] Literature review generation
- [ ] Research gap discovery
- [ ] Citation path explorer

### Phase 5 — Production Platform

- [ ] FastAPI backend
- [ ] React/Next.js frontend
- [ ] Docker deployment
- [ ] Monitoring
- [ ] CI/CD