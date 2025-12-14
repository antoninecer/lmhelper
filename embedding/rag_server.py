from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
import pickle
import requests
import time
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# --- CONFIG --------------------------------------------------------------

EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
CHAT_URL  = "http://127.0.0.1:9999/v1/chat/completions"

EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
LLM_MODEL   = "qwen2.5-7b-instruct-mlx"

ALLOWED_LANGS = ["en", "cz", "de", "pl", "it"]

INDEX = "../vectordb/faiss.index"
META  = "../vectordb/meta.pkl"

# --- LOGGING -------------------------------------------------------------

LOG_DIR  = "logs"
LOG_FILE = os.path.join(LOG_DIR, "rag.log.jsonl")

def log_interaction(payload: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    payload["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# --- LOAD VECTOR INDEX ---------------------------------------------------

index = faiss.read_index(INDEX)

with open(META, "rb") as f:
    metadata = pickle.load(f)

# --- HELPERS -------------------------------------------------------------

def embed(text):
    r = requests.post(EMBED_URL, json={
        "model": EMBED_MODEL,
        "input": text
    }).json()

    return np.array(
        r["data"][0]["embedding"],
        dtype="float32"
    ).reshape(1, -1)


def search_similar(query, k=5):
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


def deduplicate(results, limit=3):
    seen = set()
    unique = []
    for r in results:
        if r["problem"] not in seen:
            seen.add(r["problem"])
            unique.append(r)
        if len(unique) >= limit:
            break
    return unique


def ask_llm(context, query, lang="en"):
    lang_map = {
        "en": "Write answer in English. Short, technical, operational.",
        "cz": "Odpověz česky. Stručně, technicky, provozní kroky.",
        "de": "Antwort auf Deutsch. Kurz und technisch.",
        "pl": "Odpowiedz po polsku. Krótko i technicznie.",
        "it": "Rispondi in italiano. Breve e tecnico."
    }

    sys_prompt = f"""
You are an IT troubleshooting assistant.

Your task is to produce short, highly actionable runbook-style steps.
Respond with numbered bullet points.

FORMAT RULES:
- Commands must be inline, prefixed with `$`
- Do NOT use Markdown code blocks or ``` formatting
- Prefer real Linux/infra commands (lsof, df, systemctl, kubectl)
- Do NOT invent fictional tools
- Keep explanations minimal
- Mark disruptive actions clearly

{lang_map.get(lang, lang_map["en"])}
""".strip()

    user_prompt = f"""
User problem:
{query}

Relevant historical cases:
{context}

Provide root cause + recommended fix steps.
""".strip()

    start = time.time()

    r = requests.post(CHAT_URL, json={
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 200
    }).json()

    end = time.time()

    return {
        "answer": r["choices"][0]["message"]["content"],
        "response_time_seconds": round(end - start, 2)
    }

# --- API ENDPOINTS -------------------------------------------------------

@app.route("/search", methods=["POST"])
def handle_search():
    data = request.json
    query = data.get("query", "")
    lang  = data.get("lang", "en").lower()

    if lang not in ALLOWED_LANGS:
        lang = "en"

    results = deduplicate(search_similar(query))

    log_interaction({
        "mode": "search",
        "query": query,
        "lang": lang,
        "results": [
            {"problem": r["problem"], "distance": r["distance"]}
            for r in results
        ]
    })

    return jsonify(results)


@app.route("/solve", methods=["POST"])
def handle_solve():
    start_total = time.time()

    data = request.json
    query = data.get("query", "")
    lang  = data.get("lang", "en").lower()

    if lang not in ALLOWED_LANGS:
        lang = "en"

    similar_raw = search_similar(query)
    similar = deduplicate(similar_raw)

    context_text = "\n\n".join([
        f"- Problem: {x['problem']}\n"
        f"  Symptoms: {x['symptoms']}\n"
        f"  Solution: {x['solution']}"
        for x in similar
    ])

    llm_result = ask_llm(context_text, query, lang)
    total_time = round(time.time() - start_total, 2)

    log_interaction({
        "mode": "solve",
        "query": query,
        "lang": lang,
        "similar_cases": [
            {"problem": x["problem"], "distance": x["distance"]}
            for x in similar
        ],
        "answer": llm_result["answer"],
        "llm_time": llm_result["response_time_seconds"],
        "total_time": total_time,
        "model": LLM_MODEL
    })

    return jsonify({
        "llm_answer": llm_result["answer"],
        "similar_cases": similar,
        "response_time": llm_result["response_time_seconds"]
    })

# --- START SERVER --------------------------------------------------------

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001)
