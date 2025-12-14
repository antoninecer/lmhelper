lmhelper
LM Helper is an open-source tool for building a local Retrieval-Augmented Generation (RAG) system. It allows you to run a private AI assistant on your machine using local language models (LLMs) combined with a knowledge base. This setup is ideal for IT helpdesk, incident response, or any scenario where you need quick, secure answers from internal documents without relying on cloud services.
The project combines a Python backend for embedding, vector search, and RAG inference with a simple PHP frontend for user interaction. It supports vector databases like FAISS (for local, lightweight setups) or Qdrant (for more scalable options).
Features

Local LLM Integration: Use models like Qwen, Llama, or Mistral (7-8B parameters) via Apple MLX, LM Studio, or Ollama.
RAG Functionality: Ingest documents into a vector database, retrieve relevant chunks, and augment LLM prompts for accurate responses.
Private & On-Prem: All data stays local—no external APIs or data leakage.
Simple UI: PHP-based web interface for querying the AI.
Logging & Debugging: JSON logs for queries and responses.
Extensible: Easy to integrate with helpdesk tools like Zammad or Slack.

Architecture
The architecture is designed for simplicity and efficiency, running entirely on local hardware (e.g., Apple Mac mini with M-series chips for optimal MLX performance). It follows a client-server model:
High-Level Overview

Frontend (PHP): Handles user input, displays results, and calls the backend API.
Backend (Python): Manages document embedding, vector storage/search, and LLM inference.
Data Flow:
User submits a query via the PHP UI (index.php).
PHP script (call_llm.php) sends the query to the Python RAG server (rag_server.py) via HTTP or local API.
Python backend retrieves relevant documents from the vector DB (using FAISS or Qdrant), augments the prompt, and queries the local LLM.
Response is returned to PHP and displayed to the user.
Logs are stored in JSON for auditing.


Key Components

Embedding & Vector DB:
Documents (e.g., PDFs, TXT, or JSON like problems.json) are ingested and embedded into vectors.
Supports FAISS (local index) or Qdrant (server-based for larger scales).

RAG Server: Acts as the core inference engine, combining retrieval and generation.
Frontend Integration: PHP connects to the backend, configurable via lmhelper_config.php.
Diagram (ASCII representation for simplicity):

text[User] --> [PHP Frontend: index.php, call_llm.php] --> [Python Backend: rag_server.py]
                                           |
                                           v
[Vector DB: FAISS/Qdrant] <--> [Embedding: embed.py/embed_faiss.py] <--> [Knowledge Base: data/problems.json]
                                           |
                                           v
[Local LLM: via MLX/LM Studio] --> [Response] --> [Logs: rag_log.json]
For more details, see docs/architecture.md (expand on components) and docs/embedding.md (embedding process).
Scalability

Start with 1-2 Mac minis in HA (active/standby) for redundancy.
For higher loads (10+ users), scale to NVIDIA GPUs with CUDA backend.

Setup
Prerequisites

Python 3.8+ (with venv)
PHP 7+ (with web server like Apache/Nginx or built-in)
Local LLM setup (e.g., LM Studio running on localhost)
Optional: Qdrant server for advanced vector DB

Installation

Clone the repo:textgit clone https://github.com/antoninecer/lmhelper.git
cd lmhelper
Set up Python environment:textpython -m venv venv
source venv/bin/activate
pip install -r requirements.txt(Requirements include libraries like faiss-cpu, sentence-transformers, flask for server.)
Configure:
Edit lmhelper_config.php for LLM API endpoints (e.g., LM Studio URL).
Prepare knowledge base in data/ (e.g., add documents to ingest).

Embed documents:textpython embed_faiss.py  # Or embed.py for Qdrant(See qdrant_setup.md for Qdrant config.)
Start the RAG server:textpython rag_server.py
Start the PHP server:textphp -S localhost:8000Or use start.sh for automated launch.
Access the UI at http://localhost:8000/index.php.

High Availability (HA) Setup
For production:

Run 2 instances (primary + standby).
Use Nginx as reverse proxy for load balancing.
Sync vector DB via shared storage (e.g., NAS).

Usage

Ingest knowledge: Run embedding scripts with your documents.
Query via UI: Enter a problem (e.g., "How to fix network issue?") and get RAG-augmented response.
API Mode: Call /api/query on rag_server.py directly for integrations.

Example query log in rag_log.json:
JSON{"query": "Sample question", "response": "AI answer", "timestamp": "2025-12-14"}
Contributing
Fork the repo, make changes, and submit a PR. Focus on improving embedding efficiency or adding more DB backends.
License
MIT License – see LICENSE file (add if missing).
For questions, check the GitHub issues or contact the maintainer.