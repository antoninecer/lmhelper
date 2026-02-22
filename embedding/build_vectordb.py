import os
import json
import pickle
import requests
import numpy as np
import faiss
from dotenv import load_dotenv

load_dotenv()

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9999/v1").rstrip("/")
EMBED_URL = os.getenv("EMBED_URL", f"{LMSTUDIO_BASE_URL}/embeddings")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")

def embed_text(text: str) -> np.ndarray:
    r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "input": text}, timeout=60)
    r.raise_for_status()
    j = r.json()
    vec = j["data"][0]["embedding"]
    return np.array(vec, dtype="float32")

def build_input_text(row: dict) -> str:
    # Stejný styl jako doteď – kombinace polí
    parts = [
        f"Problem: {row.get('problem', '')}",
        f"Symptoms: {row.get('symptoms', '')}",
        f"Analysis: {row.get('analysis', '')}",
        f"Solution: {row.get('solution', '')}",
    ]
    return "\n".join(parts).strip()

def main():
    # např. domain=onboarding => data/onboarding/problems.jsonl, vectordb/onboarding/*
    domain = os.getenv("KB_DOMAIN", "onboarding").strip().lower()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(base_dir, ".."))

    in_file = os.path.join(repo_root, "data", domain, "problems.jsonl")
    out_dir = os.path.join(repo_root, "vectordb", domain)
    out_index = os.path.join(out_dir, "faiss.index")
    out_meta = os.path.join(out_dir, "meta.pkl")

    if not os.path.exists(in_file):
        raise FileNotFoundError(f"Input file not found: {in_file}")

    rows = []
    with open(in_file, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Invalid JSON on line {line_no}: {e}")

    if not rows:
        raise RuntimeError("No records found in JSONL")

    print(f"[INFO] Domain: {domain}")
    print(f"[INFO] Records: {len(rows)}")
    print(f"[INFO] Embedding model: {EMBED_MODEL}")

    vectors = []
    metadata = []

    for i, row in enumerate(rows, start=1):
        text = build_input_text(row)
        vec = embed_text(text)
        vectors.append(vec)
        metadata.append({
            "id": row.get("id", i),
            "problem": row.get("problem", ""),
            "symptoms": row.get("symptoms", ""),
            "analysis": row.get("analysis", ""),
            "solution": row.get("solution", ""),
        })
        if i % 5 == 0 or i == len(rows):
            print(f"[INFO] Embedded {i}/{len(rows)}")

    X = np.vstack(vectors).astype("float32")

    # L2 index (sedí na tvůj současný code a distance threshold logiku)
    index = faiss.IndexFlatL2(X.shape[1])
    index.add(X)

    os.makedirs(out_dir, exist_ok=True)
    faiss.write_index(index, out_index)
    with open(out_meta, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[OK] Saved index: {out_index}")
    print(f"[OK] Saved meta : {out_meta}")
    print(f"[OK] dim={X.shape[1]} ntotal={index.ntotal}")

if __name__ == "__main__":
    main()
