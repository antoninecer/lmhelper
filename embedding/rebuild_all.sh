#!/usr/bin/env bash
set -euo pipefail

python embedding/embed_faiss.py data/it/problems.jsonl        vectordb/it/faiss.index        vectordb/it/meta.pkl
python embedding/embed_faiss.py data/hr/problems.jsonl        vectordb/hr/faiss.index        vectordb/hr/meta.pkl
python embedding/embed_faiss.py data/finance/problems.jsonl   vectordb/finance/faiss.index   vectordb/finance/meta.pkl
python embedding/embed_faiss.py data/onboarding/problems.jsonl vectordb/onboarding/faiss.index vectordb/onboarding/meta.pkl
