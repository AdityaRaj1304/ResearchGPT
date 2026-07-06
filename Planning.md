# ResearchGPT Project Roadmap

ResearchGPT is an end-to-end Semantic Search and Retrieval-Augmented Generation (RAG) system designed to index, vectorize, and search academic literature from arXiv. The system allows users to execute natural language queries against an optimized vector corpus of cutting-edge AI research papers.

---

## Technical Stack Architecture

* **Operating System:** Linux via Ubuntu WSL 2
* **Data Processing:** Python 3.12, Pandas, PyArrow (Parquet storage)
* **Ingestion Layer:** arXiv API, lxml parser, tqdm tracking
* **Embedding Layer:** HuggingFace Sentence-Transformers (Local compute)
* **Vector Storage:** Milvus, Qdrant, or ChromaDB
* **Backend Pipeline:** Node.js, Express.js (MERN pattern synchronization)
* **Frontend Interface:** React.js

---

## Project Milestones

### Milestone 1: Problem & Planning
* **Objective:** Establish the development environment and define the engineering workflow.
* **Tasks completed:**
  * Configure Ubuntu WSL 2 as the native development environment to prevent cross-platform compilation errors.
  * Initialize a localized Linux virtual environment (`.venv`).
  * Install the base ingestion dependencies (`arxiv`, `pandas`, `pyarrow`, `tqdm`, `pyyaml`, `lxml`).
  * Construct a baseline verification pipeline (`test_fetch.py`) to confirm API connectivity and disk-write permissions.

### Milestone 2: Data Preparation & Ingestion
* **Objective:** Build a scalable, rate-limited ingestion engine to harvest and clean the core academic corpus.
* **Tasks remaining:**
  * Execute `bulk_fetch.py` to securely download the initial batch of 1,000 papers spanning `cs.CL` (Computation and Language) and `cs.LG` (Machine Learning) categories.
  * Implement pagination and graceful error-handling to survive network timeouts without losing data.
  * Clean raw LaTeX text anomalies, handle missing fields, and compress data into a snappy-compressed `.parquet` file to preserve typed structure.

### Milestone 3: Model Training, Embeddings & Vector Storage
* **Objective:** Chunk raw text data, generate dense mathematical embeddings, and index them inside a low-latency vector database.
* **Tasks remaining:**
  * Define a text chunking strategy (e.g., recursive character text splitting) to handle paper abstracts or full texts without losing contextual overlap.
  * Select and initialize a local embedding model (e.g., `sentence-transformers/all-MiniLM-L6-v2`) via PyTorch.
  * Set up a vector database instance (e.g., a local ChromaDB or a Dockerized Qdrant/Milvus container).
  * Design the vector database schema to map high-dimensional vector embeddings to their respective metadata properties (ID, title, URL, publication date).
  * Compute cosine similarity matrices during initial query testing to benchmark search accuracy.

### Milestone 4: Evaluation & RAG Integration
* **Objective:** Link the retrieval engine to a Large Language Model (LLM) and establish a validation framework for response accuracy.
* **Tasks remaining:**
  * Design a semantic retrieval function that takes a raw user query, embeds it on the fly, and pulls the top $K$ most relevant document chunks.
  * Build a prompt template that strictly binds the LLM context to the retrieved papers to prevent hallucination.
  * Connect the context window to a local or API-driven LLM backend.
  * Establish evaluation metrics (e.g., precision at $K$, retrieval latency, and context relevance checks) to verify that results are factually grounded.

### Milestone 5: Deployment & Results
* **Objective:** Connect the Python-driven data pipeline to a MERN full-stack application and deploy the user interface.
* **Tasks remaining:**
  * Spin up an Express.js/Node.js server to act as the primary API orchestration layer.
  * Connect the backend server to log query metadata, user activity, or system logs.
  * Build a clean, minimalist React.js frontend interface featuring a search query bar, a real-time progress layout, and interactive search result cards.
  * Integrate direct PDF hyperlinks (`pdf_url`) to allow users to verify raw source documentation instantly.
  * Document the final API endpoints, execution instructions, and system performance metrics inside the master README file.