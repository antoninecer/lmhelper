import json
import sys
import os
import pickle

import requests
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Config (can be overridden by .env) ----
EMBED_URL = os.getenv("EMBED_URL", "http://127.0.0.1:1234/v1/embeddings")
MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

# ---- CLI arguments ----
# Usage:
#   python embedding/embed_faiss.py data/hr/problems.jsonl vectordb/hr/faiss.index vectordb/hr/meta.pkl
DEFAULT_JSONL = os.path.join(BASE_DIR, "..", "data", "it", "problems.jsonl")
DEFAULT_INDEX = os.path.join(BASE_DIR, "..", "vectordb", "it", "faiss.index")
DEFAULT_META  = os.path.join(BASE_DIR, "..", "vectordb", "it", "meta.pkl")

JSONL = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_JSONL
INDEX = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_INDEX
META  = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_META

# Normalize relative paths (when script is run from project root)
JSONL = os.path.abspath(JSONL)
INDEX = os.path.abspath(INDEX)
META = os.path.abspath(META)

# Ensure target directories exist
os.makedirs(os.path.dirname(INDEX), exist_ok=True)
os.makedirs(os.path.dirname(META), exist_ok=True)

print(f"[INFO] JSONL    : {JSONL}")
print(f"[INFO] INDEX OUT: {INDEX}")
print(f"[INFO] META OUT : {META}")
print(f"[INFO] EMBED_URL: {EMBED_URL}")
print(f"[INFO] MODEL    : {MODEL}")

if not os.path.exists(JSONL):
    raise FileNotFoundError(f"Input JSONL not found: {JSONL}")

vectors = []
metadata = []


def build_text(row: dict) -> str:
    """
    Flexible text builder:
    - IT data usually has: problem, symptoms, analysis, solution
    - HR/Finance/Onboarding can be simpler
    """
    parts = []

    for key in ["problem", "symptoms", "analysis", "solution", "title", "question", "answer", "content"]:
        val = row.get(key)
        if val:
            parts.append(str(val).strip())

    # fallback: embed whole row if keys are different
    if not parts:
        parts.append(json.dumps(row, ensure_ascii=False))

    return "\n".join(parts).strip()


with open(JSONL, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f, start=1):
        line = line.strip()
        if not line:
            continue

        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"[WARN] Skipping invalid JSON line {idx}: {e}")
            continue

        text = build_text(row)
        if not text:
            print(f"[WARN] Skipping empty text on line {idx}")
            continue

        print(f"[{idx}] Embedding: {text[:100].replace(chr(10), ' ')}...")

        r = requests.post(
            EMBED_URL,
            json={
                "model": MODEL,
                "input": text
            },
            timeout=120
        )

        try:
            j = r.json()
        except Exception:
            raise RuntimeError(f"Embedding API returned non-JSON (line {idx}). status={r.status_code} body={r.text[:500]}")

        if r.status_code >= 400:
            raise RuntimeError(f"Embedding API error (line {idx}). status={r.status_code} body={j}")

        if "data" not in j or not j["data"]:
            raise RuntimeError(f"Embedding API bad response (line {idx}): {j}")

        vec = np.array(j["data"][0]["embedding"], dtype="float32")
        vectors.append(vec)
        metadata.append(row)

if not vectors:
    raise RuntimeError("No vectors were created (input empty or all rows invalid).")

vectors = np.vstack(vectors)

# ---- FAISS index ----
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, INDEX)

with open(META, "wb") as f:
    pickle.dump(metadata, f)

print("[OK] Saved index:", INDEX)
print("[OK] Saved meta :", META)
print(f"[OK] dim={dim} ntotal={index.ntotal}")
