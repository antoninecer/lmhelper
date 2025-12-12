from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
import pickle
import requests

app = Flask(__name__)
CORS(app)

# --- CONFIG ---
EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
MODEL_EMB = "text-embedding-nomic-embed-text-v1.5"
MODEL_LLM = "qwen2.5-7b-instruct-mlx"

INDEX = "../vectordb/faiss.index"
META = "../vectordb/meta.pkl"

# --- LOAD INDEX ---
index = faiss.read_index(INDEX)

with open(META, "rb") as f:
    metadata = pickle.load(f)


# ---- EMBEDDING FUNCTION ----
def embed(text):
    r = requests.post(EMBED_URL, json={
        "model": MODEL_EMB,
        "input": text
    }).json()

    return np.array(r["data"][0]["embedding"], dtype="float32").reshape(1, -1)


# ---- VECTOR SEARCH ----
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


# ---- ENDPOINT: SEARCH ----
@app.route("/search", methods=["POST"])
def handle_search():
    data = request.json
    query = data.get("query", "")
    return jsonify(search(query, k=3))


# ---- ENDPOINT: SOLVE (LLM + RAG) ----
@app.route("/solve", methods=["POST"])
def handle_solve():
    data = request.json
    query = data.get("query", "")

    rag_hits = search(query, k=3)

    context_text = "\n\n".join(
        f"Problem: {r['problem']}\nSymptoms: {r['symptoms']}\nAnalysis: {r['analysis']}\nSolution: {r['solution']}"
        for r in rag_hits
    )

    prompt = f"""
Uživatel hlásí problém:

"{query}"

Na základě podobných historických případů:

{context_text}

Navrhni finální analýzu a doporuč mi přesný postup, co má technik udělat.
"""

    llm_resp = requests.post(
        "http://127.0.0.1:9999/v1/chat/completions",
        json={
            "model": MODEL_LLM,
            "messages": [
                {"role": "system", "content": "Jsi zkušený IT technik a síťař."},
                {"role": "user", "content": prompt}
            ]
        }
    ).json()

    return jsonify({
        "matches": rag_hits,
        "final_answer": llm_resp["choices"][0]["message"]["content"]
    })


# ---- START SERVER ----
app.run(host="127.0.0.1", port=5001)
