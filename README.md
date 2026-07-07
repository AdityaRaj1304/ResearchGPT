ResearchGPT

Autonomous Academic Research & Synthesis Engine

ResearchGPT is a production-oriented Retrieval-Augmented Generation (RAG) system designed for academic research. Unlike conventional RAG systems that rely solely on vector similarity, ResearchGPT implements a multi-stage retrieval architecture combining dense retrieval, sparse retrieval, and neural reranking to deliver highly relevant, citation-grounded responses.

Features

Hybrid Retrieval Pipeline

Combines:

Dense Semantic Search (ChromaDB + MiniLM Embeddings)

Sparse Lexical Search (BM25)

This enables retrieval of both:

Semantically relevant papers

Exact technical terminology

Neural Reranking

Uses:

cross-encoder/ms-marco-MiniLM-L-6-v2

The reranker performs deep query-document attention and reorders candidate papers retrieved from the hybrid search stage.

Pipeline:

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


Citation-Grounded Generation

Responses are generated using retrieved academic sources and include paper references to reduce hallucinations.

Local-First Architecture

Runs completely on local hardware using:

Ollama

Llama 3

ChromaDB

No external API calls required.

Current Architecture

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


Performance Benchmarks

An automated 25-query academic benchmark suite validates the two-stage retrieval architecture. The results below demonstrate the classic precision-recall trade-off when moving from Dense to Hybrid, and the ultimate precision recovery achieved by Cross-Encoder reranking.

Architecture Strategy

Precision@5

Recall@5

Recall@10

MRR

Dense Only

0.5360

0.1151

0.1588

0.8100

Hybrid (Dense + Sparse)

0.4000

0.1107

0.1879

0.6124

Hybrid + Reranker

0.6640

0.1687

0.2480

0.8647

Analysis: While raw Hybrid search increased candidate recall (Recall@10: 0.1879) by casting a wider keyword net, it introduced noise that temporarily lowered top-level precision. The Stage 2 MS-MARCO Cross-Encoder successfully filtered this noise, resulting in a +23.8% relative gain in Precision@5 over the baseline and near-perfect first-rank relevance (MRR: 0.8647).

Dataset

Current Corpus:

1000+ Research Papers

Domains:

Machine Learning

Natural Language Processing

Metadata Stored:

Paper ID

Title

Abstract

Authors

Publication Date

Categories

PDF URL

Source:

arXiv API

Tech Stack

Retrieval: ChromaDB, rank-bm25

Embeddings: all-MiniLM-L6-v2 (ONNX)

Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2

LLM Provider: Ollama (Llama 3)

Backend: Python 3.12+

Quick Start

1. Prerequisites

Ensure Ollama is installed and running locally. Pull the model:

ollama pull llama3


2. Setup

Clone the repository and install dependencies:

python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -r requirements.txt


3. Build the Database

Run the embedding pipeline to index the parquet dataset:

python src/models/build_embeddings.py


4. Synthesize Answers

Run the master engine to interact with your research corpus:

python src/models/generate_answer.py


Repository Structure

ResearchGPT/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── vector_store/
│   └── evaluation/
│
├── src/
│   ├── data/
│   ├── models/
│   └── evaluation/
│
├── requirements.txt
│
└── README.md


Roadmap

Phase 1 — Core RAG

[x] arXiv ingestion

[x] Embedding generation

[x] ChromaDB indexing

[x] BM25 retrieval

[x] Hybrid search

[x] Cross-encoder reranking

[x] Citation-based answering

Phase 2 — Evaluation Framework

[x] Precision@K

[x] Recall@K

[x] MRR

[ ] nDCG

[x] Retrieval benchmarking

Phase 3 — Full Paper Intelligence

[ ] PDF parsing

[ ] Section-aware chunking

[ ] Citation graph

[ ] Related paper recommendations

Phase 4 — Research Intelligence

[ ] Paper comparison

[ ] Literature review generation

[ ] Research gap discovery

[ ] Citation path explorer

Phase 5 — Production Platform

[ ] FastAPI backend

[ ] React/Next.js frontend

[ ] Docker deployment

[ ] Monitoring

[ ] CI/CD

Built by Aditya Raj Gupta | ResearchGPT Evolution Cycle 2026