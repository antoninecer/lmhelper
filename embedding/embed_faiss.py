import json
import requests
import faiss
import numpy as np
import os
import pickle

EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
MODEL = "text-embedding-nomic-embed-text-v1.5"

JSONL = "../data/problems.jsonl"

# Where to store FAISS DB
os.makedirs("../vectordb", exist_ok=True)
META = "../vectordb/meta.pkl"
INDEX = "../vectordb/faiss.index"

vectors = []
metadata = []

with open(JSONL, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        row = json.loads(line)

        # Our dataset has only: problem + solution
        text = f"{row['problem']} {row['solution']}"

        print(f"[{idx}] Embedding:", text[:80], "...")

        r = requests.post(EMBED_URL, json={
            "model": MODEL,
            "input": text
        }).json()

        vec = np.array(r["data"][0]["embedding"], dtype="float32")
        vectors.append(vec)
        metadata.append(row)

vectors = np.vstack(vectors)

# FAISS index
dim = vectors.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(vectors)

faiss.write_index(index, INDEX)

with open(META, "wb") as f:
    pickle.dump(metadata, f)

print("DONE — FAISS index and metadata stored successfully!")
print("Vector shape:", vectors.shape)

