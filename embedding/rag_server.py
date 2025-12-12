from flask import Flask, request, jsonify
import faiss
import numpy as np
import pickle
import requests

app = Flask(__name__)

EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
MODEL = "text-embedding-nomic-embed-text-v1.5"

INDEX = "../vectordb/faiss.index"
META = "../vectordb/meta.pkl"

index = faiss.read_index(INDEX)

with open(META, "rb") as f:
    metadata = pickle.load(f)

def embed(text):
    r = requests.post(EMBED_URL, json={
        "model": MODEL,
        "input": text
    }).json()
    return np.array(r["data"][0]["embedding"], dtype="float32").reshape(1, -1)

def search(query, k=3):
    vec = embed(query)
    distances, indices = index.search(vec, k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = metadata[idx]
        results.append({
            "distance": float(dist),
            "problem": row["problem"],
            "symptoms": row["symptoms"],
            "analysis": row["analysis"],
            "solution": row["solution"]
        })

    return results

@app.route("/search", methods=["POST"])
def handle_search():
    data = request.json
    query = data.get("query", "")
    results = search(query, k=3)
    return jsonify(results)

app.run(host="127.0.0.1", port=5001)

