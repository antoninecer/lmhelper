import json
import faiss
import numpy as np
import requests
import pickle

EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
MODEL = "text-embedding-nomic-embed-text-v1.5"

# Correct FAISS paths
INDEX = "../vectordb/faiss.index"
META = "../vectordb/meta.pkl"

# Load index
index = faiss.read_index(INDEX)

# Load metadata
with open(META, "rb") as f:
    metadata = pickle.load(f)

def embed(text):
    r = requests.post(EMBED_URL, json={
        "model": MODEL,
        "input": text
    }).json()

    vec = np.array(r["data"][0]["embedding"], dtype="float32")
    return vec.reshape(1, -1)

def search(query, k=3):
    vec = embed(query)
    distances, indices = index.search(vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = metadata[idx]
        results.append({
            "distance": float(dist),
            "problem": row["problem"],
            "solution": row["solution"]
        })

    return results

if __name__ == "__main__":
    q = input("Zadej dotaz: ")
    out = search(q, k=3)
    print(json.dumps(out, indent=2, ensure_ascii=False))

