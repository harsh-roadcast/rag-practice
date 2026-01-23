# 🚀 AWS EC2 RAG Pipeline (Hybrid Search)

An advanced Retrieval-Augmented Generation (RAG) system built to handle high-density technical documentation. This project leverages Elasticsearch, LangChain, and Hybrid Search (Vector + BM25) to provide high-accuracy answers from the AWS EC2 User Guide.
📊 Performance Summary

After extensive benchmarking across multiple chunking strategies and retrieval methods, this pipeline achieved:

    Hit Rate: 84.00%

    MRR (Mean Reciprocal Rank): 0.709

    Avg Latency: ~0.05s

## 🏗️ Project Architecture

The system utilizes a multi-stage retrieval process to overcome the "vocabulary gap" found in pure semantic search.

    Ingestion: PDF processing → Recursive Character Splitting → Multi-vector indexing.

    Storage: * Elasticsearch Store: Dense vector embeddings.

        BM25 Retriever: Sparse keyword indexing for technical specifics.

    Retrieval: Parallel execution of Semantic and Keyword search.

    Ranking: Reciprocal Rank Fusion (RRF) to merge and prioritize results.

## 🛠️ Tech Stack

    Orchestration: LangChain

    Database: Elasticsearch (via Docker)

    Embeddings: sentence-transformers/all-mpnet-base-v2

    API Framework: FastAPI

    Environment: Python 3.13 + Poetry

## 🚀 Getting Started
    1. Prerequisites

    Docker & Docker Compose

    Python 3.12+

    2. Setup Environment
## Bash
    git init
## Clone the repository
    git clone https://github.com/harsh-roadcast/rag-practice.git
    cd rag-practice

### Install dependencies
poetry install

3. Spin up Elasticsearch
Bash

docker-compose up -d

4. Run the Pipeline
Bash

# Ingest the documentation
python app/services/ingest.py

# Run the Hybrid Search benchmark
python test/test_hybrid.py

📈 Benchmarking Results

We conducted a head-to-head comparison between standard Vector Search and our optimized Hybrid approach using a "Gold Standard" dataset of 50 queries.
Metric	Vector (Base)	Hybrid (Optimized)	Delta
Hit Rate	68.00%	84.00%	+16%
MRR	0.621	0.709	+0.088
Latency	0.058s	0.052s	-0.006s

Conclusion: Hybrid search is strictly superior for technical documentation, specifically improving retrieval for CLI commands, instance types (e.g., t3.micro), and error codes.
📂 Project Structure
Plaintext

rag-practice/
├── app/
│   ├── core/           # Chunking & Embedding logic
│   ├── services/       # RAG Pipeline & Vector DB management
│   └── main.py         # FastAPI Entrypoint
├── benchmark/          # Gold standard datasets & evaluation scripts
├── data/
│   ├── manuals/        # Source PDFs
│   └── output/         # Processed JSON chunks (Git Ignored)
├── test/               # Integration and Hybrid tests
└── docker-compose.yml  # Elasticsearch & Kibana setup

📝 License

Distributed under the MIT License.

Would you like me to add a section on how to use the API endpoints specifically, or does this cover the core project structure well enough?