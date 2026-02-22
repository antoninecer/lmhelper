from flask import Flask, request, jsonify
from flask_cors import CORS
import faiss
import numpy as np
import pickle
import requests
import time
import json
import re
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:9999/v1").rstrip("/")

EMBED_URL = os.getenv("EMBED_URL", f"{LMSTUDIO_BASE_URL}/embeddings")
CHAT_URL  = os.getenv("CHAT_URL",  f"{LMSTUDIO_BASE_URL}/chat/completions")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_MODEL   = os.getenv("LLM_MODEL",   "qwen2.5-7b-instruct-mlx")

ALLOWED_LANGS = ["en", "cz", "de", "pl", "it"]

# Similarity filtering (FAISS "distance": lower = closer match)
SIMILAR_MAX_DIST = float(os.getenv("SIMILAR_MAX_DIST", "0.85"))
SIMILAR_K_SOLVE  = int(os.getenv("SIMILAR_K_SOLVE", "5"))
SIMILAR_K_ZAMMAD = SIMILAR_K_SOLVE

# Paths robust vůči working directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(BASE_DIR, "..", "vectordb", "faiss.index")
META  = os.path.join(BASE_DIR, "..", "vectordb", "meta.pkl")

# ----------------------------------------------------------------------
# ZAMMAD CONFIG
# ----------------------------------------------------------------------

ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://127.0.0.1:8080")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")
ZAMMAD_LANG = os.getenv("ZAMMAD_LANG", "en").lower()

if ZAMMAD_LANG not in ALLOWED_LANGS:
    ZAMMAD_LANG = "en"

if not ZAMMAD_TOKEN:
    raise RuntimeError("ZAMMAD_TOKEN is not set (check .env or environment variables)")

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------

LOG_DIR  = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag.log.jsonl")

def log_interaction(payload: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    payload["timestamp"] = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------
# LOAD VECTOR INDEX
# ----------------------------------------------------------------------

index = faiss.read_index(INDEX)

with open(META, "rb") as f:
    metadata = pickle.load(f)

#FAISS_METRIC = getattr(index, "metric_type", faiss.METRIC_L2)
#IS_IP = (FAISS_METRIC == faiss.METRIC_INNER_PRODUCT)
#print("FAISS metric_type:", FAISS_METRIC, "IS_IP:", IS_IP, "d:", index.d, "ntotal:", index.ntotal, "meta:", len(metadata))

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------


def normalize_numbered_steps(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps = []
    for l in lines:
        m = re.match(r"^\s*\d+\.\s*(.*)$", l)
        if not m:
            continue
        body = m.group(1).strip()
        # když je to jen příkaz, přilep ho k předchozímu kroku
        if body.startswith("$") and steps:
            steps[-1] = steps[-1].rstrip() + " " + body
        else:
            steps.append(body)

    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))

def filter_similar(similar):
    similar = sorted(similar, key=lambda x: x["distance"])
    return [c for c in similar if c["distance"] <= SIMILAR_MAX_DIST]

def embed(text: str) -> np.ndarray:
    r = requests.post(EMBED_URL, json={
        "model": EMBED_MODEL,
        "input": text
    })

    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Embedding API returned non-JSON. status={r.status_code} body={r.text[:500]}")

    if r.status_code >= 400:
        raise RuntimeError(f"Embedding API error. status={r.status_code} body={j}")

    if "data" not in j or not j["data"]:
        raise RuntimeError(f"Embedding API bad response: {j}")

    return np.array(j["data"][0]["embedding"], dtype="float32").reshape(1, -1)


def search_similar(query: str, k: int = 5):
    vec = embed(query)
    distances, indices = index.search(vec, k)

    print("RAW D:", distances[0])
    print("RAW I:", indices[0])


    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = metadata[int(idx)]
        results.append({
            "id": int(row.get("id", idx)),   # primárně reálné ID z JSONL
            "distance": float(dist),
            "problem": row.get("problem", ""),
            "symptoms": row.get("symptoms", ""),
            "analysis": row.get("analysis", ""),
            "solution": row.get("solution", "")
        })
    return results


def deduplicate(results, limit: int = 3):
    seen = set()
    unique = []
    for r in results:
        key = r.get("problem", "")
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= limit:
            break
    return unique


def distance_label(dist: float, lang: str = "en") -> str:
    labels = {
        "en": {
            "almost_same": "almost the same problem",
            "very_similar": "very similar incident",
            "related": "generally related topic",
            "noise": "likely noise / weak match",
        },
        "cz": {
            "almost_same": "téměř stejný problém",
            "very_similar": "velmi podobný incident",
            "related": "obecně příbuzné téma",
            "noise": "spíš šum / slabá shoda",
        },
        "de": {
            "almost_same": "fast identisches Problem",
            "very_similar": "sehr ähnlicher Vorfall",
            "related": "allgemein verwandtes Thema",
            "noise": "eher Rauschen / schwache Übereinstimmung",
        },
        "pl": {
            "almost_same": "prawie ten sam problem",
            "very_similar": "bardzo podobny incydent",
            "related": "ogólnie powiązany temat",
            "noise": "raczej szum / słabe dopasowanie",
        },
        "it": {
            "almost_same": "quasi lo stesso problema",
            "very_similar": "incidente molto simile",
            "related": "tema generalmente correlato",
            "noise": "probabile rumore / corrispondenza debole",
        },
    }

    d = labels.get(lang, labels["en"])

    if dist < 0.80:
        return d["almost_same"]
    if dist < 0.88:
        return d["very_similar"]
    if dist < 0.95:
        return d["related"]
    return d["noise"]


def enrich_similar_cases(similar_cases, lang: str):
    for i, x in enumerate(similar_cases, start=1):
        x["distance_label"] = distance_label(x["distance"], lang)
        x["case_id"] = x["id"]
        x["label"] = x["distance_label"]
        x["rank"] = i
    return similar_cases

def get_similar_cases(query: str, lang: str, k: int, dedup_limit: int = 3):
    raw = search_similar(query, k=k)          # vrací v pořadí FAISS (většinou už best-first)
    raw = filter_similar(raw)                 # seřadí + ořízne podle SIMILAR_MAX_DIST
    raw = deduplicate(raw, limit=dedup_limit) # zachová best-first
    raw = enrich_similar_cases(raw, lang)     # rank sedí
    return raw

def build_context_text(similar_cases) -> str:
    # Kontext pro LLM – jen “Similar incidents”
    parts = []
    for x in similar_cases:
        parts.append(
            f"- Similar incident [ID {x['id']}] (dist {x['distance']:.3f}, {x['distance_label']}):\n"
            f"  Problem: {x['problem']}\n"
            f"  Symptoms: {x['symptoms']}\n"
            f"  Resolution: {x['solution']}"
        )
    return "\n\n".join(parts)


def format_similar_lines(similar_cases) -> list[str]:
    """
    Hotový výpis pro UI/Zammad: vždy používá DB ID, distance i interpretaci.
    """
    lines = []
    for x in similar_cases:
        lines.append(
            f"ID {x['id']} ({x['distance']:.3f}, {x['distance_label']}): {x['problem']}"
        )
    return lines


def ask_llm(context: str, query: str, lang: str = "en"):
    lang_map = {
        "en": "Write answer in English. Short, technical, operational.",
        "cz": "Odpověz česky. Stručně, technicky, provozní kroky.",
        "de": "Antwort auf Deutsch. Kurz und technisch.",
        "pl": "Odpowiedz po polsku. Krótko i technicznie.",
        "it": "Rispondi in italiano. Breve e tecnico."
    }
    lang_instruction = lang_map.get(lang, lang_map["en"])

    # ---- LLM params z .env ----
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens  = int(os.getenv("LLM_MAX_TOKENS", "220"))
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    # ---- sys_prompt z .env (varianta A: \n + {lang_instruction}) ----
    sys_tmpl = (os.getenv("LLM_SYS_PROMPT", "") or "").strip()
    if not sys_tmpl:
        # fallback, kdyby nebyl .env
        sys_tmpl = (
            "You are an IT operations troubleshooting assistant.\n\n"
            "Your task is to produce short, highly actionable, runbook-style remediation steps.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Output MUST contain numbered steps only (each line starts with \"1.\", \"2.\", ...)\n"
            "- Each step must be on its own line (newline-separated)\n"
            "- Do NOT output any standalone summary lines, headers, or sections\n"
            "- Commands must be inline and prefixed with `$`\n"
            "- Do NOT use markdown, code blocks, or ``` formatting\n"
            "- Prefer real Linux / infra commands\n"
            "- Do NOT invent fictional tools or commands\n"
            "- Keep explanations minimal and technical\n"
            "- Clearly mark disruptive actions (restart, kill, delete)\n"
            "{lang_instruction}"
        )

    # z .env dostaneš \n jako text -> převést na reálné nové řádky
    sys_tmpl = sys_tmpl.replace("\\n", "\n")

    # bezpečně dosadit jen pokud tam placeholder je
    if "{lang_instruction}" in sys_tmpl:
        sys_prompt = sys_tmpl.format(lang_instruction=lang_instruction).strip()
    else:
        sys_prompt = (sys_tmpl + "\n" + lang_instruction).strip()

    # ---- user prompt template volitelně z .env ----
    user_tmpl = (os.getenv("LLM_USER_PROMPT", "") or "").strip()
    if not user_tmpl:
        user_tmpl = (
            "User problem:\n{query}\n\n"
            "Similar historical incidents (if any):\n{context}\n\n"
            "Provide root cause + recommended fix steps."
        )
    user_tmpl = user_tmpl.replace("\\n", "\n")

    # stejně bezpečně jako u sys_prompt
    if "{query}" in user_tmpl or "{context}" in user_tmpl:
        user_prompt = user_tmpl.format(query=query, context=context).strip()
    else:
        user_prompt = f"{query}\n\n{context}".strip()

    start = time.time()

    r = requests.post(
        CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        },
        timeout=timeout_sec
    )

    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Chat API returned non-JSON. status={r.status_code} body={r.text[:500]}")

    if r.status_code >= 400:
        raise RuntimeError(f"Chat API error. status={r.status_code} body={j}")

    if "choices" not in j or not j["choices"]:
        raise RuntimeError(f"Chat API bad response: {j}")

    raw_answer = j["choices"][0]["message"]["content"] or ""
    fixed_answer = normalize_numbered_steps(raw_answer)

    end = time.time()
    return {
        "answer": fixed_answer,
        "raw_answer": raw_answer,  # nech si pro debug (klidně pak smaž)
        "response_time_seconds": round(end - start, 2)
    }

def zammad_post_internal_note(ticket_id: int, text: str):
    url = f"{ZAMMAD_URL}/api/v1/ticket_articles"
    headers = {
        "Authorization": f"Token token={ZAMMAD_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "ticket_id": ticket_id,
        "body": text,
        "type": "note",
        "internal": True
    }

    r = requests.post(url, headers=headers, json=payload)

    print("=== ZAMMAD WRITE BACK ===")
    print("Status:", r.status_code)
    print("Response:", r.text)


def format_zammad_note(llm_answer: str, similar_cases) -> str:
    if not similar_cases:
        return llm_answer.strip()

    header_lines = ["Similar historical incidents used:"]
    header_lines.extend([f"- {line}" for line in format_similar_lines(similar_cases)])

    return "\n".join(header_lines) + "\n\n" + llm_answer.strip()

# ----------------------------------------------------------------------
# ZAMMAD WEBHOOK DEDUP (proti dvojímu spuštění triggeru)
# ----------------------------------------------------------------------

PROCESSED = {}  # key -> timestamp
DEDUP_WINDOW_SEC = int(os.getenv("ZAMMAD_DEDUP_WINDOW_SEC", "90"))

def seen_recently(key: str) -> bool:
    now = time.time()
    for k in list(PROCESSED.keys()):
        if now - PROCESSED[k] > DEDUP_WINDOW_SEC:
            del PROCESSED[k]
    if key in PROCESSED:
        return True
    PROCESSED[key] = now
    return False

def run_pipeline(query: str, lang: str, k: int, dedup_limit: int = 3):
    llm_result = ask_llm("", query, lang)  # primární odpověď bez DB
    similar = get_similar_cases(query, lang, k=k, dedup_limit=dedup_limit)
    return llm_result, similar

# ----------------------------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/search", methods=["POST"])
def handle_search():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    lang  = (data.get("lang") or "en").lower()

    if lang not in ALLOWED_LANGS:
        lang = "en"

    results = get_similar_cases(query, lang, k=SIMILAR_K_SOLVE, dedup_limit=3)

    log_interaction({
        "mode": "search",
        "query": query,
        "lang": lang,
        "similar_cases": [
            {"id": x["id"], "problem": x["problem"], "distance": x["distance"], "label": x["distance_label"]}
            for x in results
        ],
        "model": LLM_MODEL
    })

    return jsonify(results)


@app.route("/solve", methods=["POST"])
def handle_solve():
    start_total = time.time()

    data = request.json or {}
    query = (data.get("query") or "").strip()
    lang  = (data.get("lang") or "en").lower()

    if lang not in ALLOWED_LANGS:
        lang = "en"

    llm_result, similar = run_pipeline(query, lang, k=SIMILAR_K_SOLVE, dedup_limit=3)
    total_time = round(time.time() - start_total, 2)

    # hotový text pro UI (když frontend ignoruje id/label v JSONu)
    similar_lines = format_similar_lines(similar) if similar else []
    similar_text = "\n".join(similar_lines) if similar_lines else ""

    log_interaction({
        "mode": "solve",
        "query": query,
        "lang": lang,
        "similar_cases": [
            {"id": x["id"], "problem": x["problem"], "distance": x["distance"], "label": x["distance_label"]}
            for x in similar
        ],
        "answer": llm_result["answer"],
        "llm_time": llm_result["response_time_seconds"],
        "total_time": total_time,
        "model": LLM_MODEL
    })

    return jsonify({
        "llm_answer": llm_result["answer"],
        "similar_cases": similar,                 # id + distance + distance_label + rank + case_id + label
        "similar_cases_lines": similar_lines,     # <- hotové řádky (ID + dist + label + problem)
        "similar_cases_text": similar_text,       # <- hotový blok (když chceš jen printnout)
        "response_time": llm_result["response_time_seconds"],
        "total_time": total_time
    })


@app.route("/zammad", methods=["POST"])
def zammad_webhook():
    payload = request.json or {}

    print("=== ZAMMAD WEBHOOK RECEIVED ===")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:5000])

    try:
        ticket = payload.get("ticket", {}) or {}
        article = payload.get("article", {}) or {}

        ticket_id = ticket.get("id")
        article_id = article.get("id")

        if not ticket_id:
            return jsonify({"status": "error", "error": "missing ticket.id"}), 400

        dedup_key = f"{ticket_id}:{article_id}" if article_id else f"{ticket_id}:{payload.get('event_id', 'no_event')}"
        if seen_recently(dedup_key):
            return jsonify({"status": "ok", "dedup": True})

        title = (ticket.get("title") or "").strip()
        body = (article.get("body") or "").strip()
        query = f"{title}\n\n{body}".strip()

        print("=== QUERY SENT TO LLM ===")
        print(query)

        start_total = time.time()
        llm_result, similar = run_pipeline(query, ZAMMAD_LANG, k=SIMILAR_K_SOLVE, dedup_limit=3)
        total_time = round(time.time() - start_total, 2)

        answer_note = format_zammad_note(llm_result["answer"], similar)

        print("=== LLM ANSWER ===")
        print(answer_note)

        zammad_post_internal_note(ticket_id, answer_note)

        log_interaction({
            "mode": "zammad",
            "ticket_id": ticket_id,
            "article_id": article_id,
            "lang": ZAMMAD_LANG,
            "query_preview": query[:500],
            "similar_cases": [
                {"id": x["id"], "problem": x["problem"], "distance": round(x["distance"], 4), "label": x["distance_label"]}
                for x in similar
            ],
            "answer": answer_note,
            "llm_time": llm_result["response_time_seconds"],
            "total_time": total_time,
            "model": LLM_MODEL
        })

        return jsonify({"status": "ok"})

    except Exception as e:
        print("ERROR:", e)
        log_interaction({
            "mode": "zammad_error",
            "error": str(e),
            "payload_preview": json.dumps(payload, ensure_ascii=False)[:1000]
        })
        return jsonify({"status": "error", "error": str(e)}), 500


# ----------------------------------------------------------------------
# START SERVER
# ----------------------------------------------------------------------

if __name__ == "__main__":
		host = os.getenv("RAG_HOST", "127.0.0.1")
		port = int(os.getenv("RAG_PORT", "5001"))
		app.run(host=host, port=port)    

