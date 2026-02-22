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
from datetime import datetime, UTC
from dotenv import load_dotenv
from typing import Dict, List, Tuple, Optional

load_dotenv()

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
EMBED_URL = os.getenv("EMBED_URL", f"{LMSTUDIO_BASE_URL}/embeddings")
CHAT_URL = os.getenv("CHAT_URL", f"{LMSTUDIO_BASE_URL}/chat/completions")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-4b-2507")

SIMILAR_MAX_DIST = float(os.getenv("SIMILAR_MAX_DIST", "0.84"))
SIMILAR_K_SOLVE = int(os.getenv("SIMILAR_K_SOLVE", "5"))
SIMILAR_K_ZAMMAD = int(os.getenv("SIMILAR_K_ZAMMAD", str(SIMILAR_K_SOLVE)))

DEFAULT_DOMAIN = (os.getenv("DEFAULT_DOMAIN", "it") or "it").strip().lower()
DEFAULT_REPLY_LANG = (os.getenv("DEFAULT_REPLY_LANG", "cs") or "cs").strip().lower()
DEFAULT_SEARCH_LANG = (os.getenv("DEFAULT_SEARCH_LANG", "en") or "en").strip().lower()

INTERNAL_EMAIL_DOMAIN = (os.getenv("INTERNAL_EMAIL_DOMAIN", "ventureout.cz") or "ventureout.cz").strip().lower()

ALLOWED_LANGS = {"cs", "en", "de", "pl", "it"}
LANG_ALIASES = {"cz": "cs"}

# Zammad
ZAMMAD_URL = (os.getenv("ZAMMAD_URL", "") or "").rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN", "").strip()

# Povolené skupiny (názvy v Zammadu)
AI_ENABLED_GROUPS = {
    x.strip().lower()
    for x in (os.getenv("AI_ENABLED_GROUPS", "IT,HR,Finance,Onboarding") or "").split(",")
    if x.strip()
}

# Group name -> domain map
GROUP_NAME_TO_DOMAIN = {
    (os.getenv("GROUP_MAP_IT", "IT") or "IT").strip().lower(): "it",
    (os.getenv("GROUP_MAP_HR", "HR") or "HR").strip().lower(): "hr",
    (os.getenv("GROUP_MAP_FINANCE", "Finance") or "Finance").strip().lower(): "finance",
    (os.getenv("GROUP_MAP_ONBOARDING", "Onboarding") or "Onboarding").strip().lower(): "onboarding",
}

# ----------------------------------------------------------------------
# KB CONFIG (4 domény)
# ----------------------------------------------------------------------

def env_path(name: str, default_rel: str) -> str:
    val = (os.getenv(name, default_rel) or default_rel).strip()
    if os.path.isabs(val):
        return val
    return os.path.abspath(os.path.join(ROOT_DIR, val))

KB_CONFIG = {
    "it": {
        "jsonl": env_path("KB_IT_JSONL", "data/it/problems.jsonl"),
        "index": env_path("KB_IT_INDEX", "vectordb/it/faiss.index"),
        "meta": env_path("KB_IT_META", "vectordb/it/meta.pkl"),
        "search_lang": (os.getenv("IT_SEARCH_LANG", DEFAULT_SEARCH_LANG) or DEFAULT_SEARCH_LANG).strip().lower(),
    },
    "hr": {
        "jsonl": env_path("KB_HR_JSONL", "data/hr/problems.jsonl"),
        "index": env_path("KB_HR_INDEX", "vectordb/hr/faiss.index"),
        "meta": env_path("KB_HR_META", "vectordb/hr/meta.pkl"),
        "search_lang": (os.getenv("HR_SEARCH_LANG", DEFAULT_SEARCH_LANG) or DEFAULT_SEARCH_LANG).strip().lower(),
    },
    "finance": {
        "jsonl": env_path("KB_FINANCE_JSONL", "data/finance/problems.jsonl"),
        "index": env_path("KB_FINANCE_INDEX", "vectordb/finance/faiss.index"),
        "meta": env_path("KB_FINANCE_META", "vectordb/finance/meta.pkl"),
        "search_lang": (os.getenv("FINANCE_SEARCH_LANG", DEFAULT_SEARCH_LANG) or DEFAULT_SEARCH_LANG).strip().lower(),
    },
    "onboarding": {
        "jsonl": env_path("KB_ONBOARDING_JSONL", "data/onboarding/problems.jsonl"),
        "index": env_path("KB_ONBOARDING_INDEX", "vectordb/onboarding/faiss.index"),
        "meta": env_path("KB_ONBOARDING_META", "vectordb/onboarding/meta.pkl"),
        "search_lang": (os.getenv("ONBOARDING_SEARCH_LANG", DEFAULT_SEARCH_LANG) or DEFAULT_SEARCH_LANG).strip().lower(),
    },
}

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag.log.jsonl")

def log_interaction(payload: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    payload["timestamp"] = datetime.now(UTC).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------
# LOAD VECTOR DBS
# ----------------------------------------------------------------------

KB_RUNTIME: Dict[str, Dict] = {}

def load_kbs():
    for domain, cfg in KB_CONFIG.items():
        idx_path = cfg["index"]
        meta_path = cfg["meta"]

        if not os.path.exists(idx_path):
            print(f"[KB][WARN] Missing index for {domain}: {idx_path}")
            continue
        if not os.path.exists(meta_path):
            print(f"[KB][WARN] Missing meta for {domain}: {meta_path}")
            continue

        index = faiss.read_index(idx_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        KB_RUNTIME[domain] = {
            "index": index,
            "metadata": metadata,
            "cfg": cfg,
        }
        print(f"[KB] Loaded {domain}: {idx_path}")

load_kbs()

if not KB_RUNTIME:
    raise RuntimeError("No KB loaded. Check vectordb/* paths in .env")

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def normalize_lang(lang: str) -> str:
    x = (lang or "").strip().lower()
    x = LANG_ALIASES.get(x, x)
    return x if x in ALLOWED_LANGS else DEFAULT_REPLY_LANG

def safe_json_loads(s: str) -> Optional[dict]:
    try:
        return json.loads(s)
    except Exception:
        return None

def normalize_numbered_steps(text: str) -> str:
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    steps = []

    for l in lines:
        m = re.match(r"^\s*(\d+)[\.\)]\s*(.*)$", l)
        if m:
            body = m.group(2).strip()
            if body:
                if body.startswith("$") and steps:
                    steps[-1] = steps[-1].rstrip() + " " + body
                else:
                    steps.append(body)
            continue

        # fallback: pokud model nevrátil číslování, zkus aspoň použít nečíslované řádky
        if l and not l.startswith("{") and not l.startswith("```"):
            steps.append(l)

    # dedup krátkých duplicit
    uniq = []
    seen = set()
    for s in steps:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            uniq.append(s)

    return "\n".join(f"{i}. {s}" for i, s in enumerate(uniq, start=1))

def post_chat(messages: List[dict], temperature: float = 0.2, max_tokens: int = 400, top_p: float = 1.0, timeout_sec: float = 60):
    r = requests.post(
        CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
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

    return (j["choices"][0]["message"]["content"] or "").strip()

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

# ----------------------------------------------------------------------
# ROUTING (AI-first, keyword fallback)
# ----------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "it": [
        "email", "vpn", "login", "heslo", "password", "server", "síť", "network", "pc", "notebook",
        "mail", "outlook", "m365", "printer", "tiskárna", "ssh", "dns", "wifi", "account"
    ],
    "hr": [
        "zaměstnanec", "dovolen", "výplata", "mzda", "pracovní", "pracownik", "urlop", "wynagrodzenie",
        "nemoc", "sick", "leave", "parental", "mateřská", "otcovská", "pracovní doba", "pauza"
    ],
    "finance": [
        "faktura", "invoice", "platba", "payment", "dodavatel", "vendor", "upomínka", "účetnictví",
        "po", "purchase order", "dpH", "vat", "bank", "banka", "cash", "úhrada"
    ],
    "onboarding": [
        "nástup", "nastupuje", "onboarding", "nový kolega", "new employee", "first day", "equipment",
        "vybavení", "access", "přístupy", "účet pro nového", "welcome"
    ],
}

def keyword_fallback_domain(query: str) -> Tuple[str, str]:
    q = (query or "").lower()
    scores = {d: 0 for d in DOMAIN_KEYWORDS.keys()}
    for d, kws in DOMAIN_KEYWORDS.items():
        for kw in kws:
            if kw in q:
                scores[d] += 1
    best = max(scores, key=scores.get)
    if scores[best] == 0:
        return DEFAULT_DOMAIN, "fallback_default_domain"
    return best, f"keyword_classifier:{scores}"

def detect_domain_ai(query: str) -> Tuple[str, str]:
    """
    Vrací (domain, reason). AI klasifikace -> JSON.
    Fallback na keyword pokud model vrátí blbost.
    """
    router_sys = (
        "You classify internal helpdesk requests into exactly one domain.\n"
        "Allowed domains: it, hr, finance, onboarding.\n"
        "Return ONLY JSON object with keys: domain, reason.\n"
        "domain must be exactly one of: it, hr, finance, onboarding.\n"
        "Examples:\n"
        '{"domain":"it","reason":"email/VPN/login access issue"}\n'
        '{"domain":"hr","reason":"employee leave/payroll/labor process"}\n'
        '{"domain":"finance","reason":"invoice/payment/vendor/accounting process"}\n'
        '{"domain":"onboarding","reason":"new employee setup/access/equipment"}'
    )
    router_user = f"Request:\n{query}"

    try:
        raw = post_chat(
            [{"role": "system", "content": router_sys}, {"role": "user", "content": router_user}],
            temperature=0.0,
            max_tokens=80,
            top_p=1.0,
            timeout_sec=20
        )
        parsed = safe_json_loads(raw)
        if parsed and parsed.get("domain") in {"it", "hr", "finance", "onboarding"}:
            return parsed["domain"], f"ai_router:{parsed.get('reason', '').strip()}"
    except Exception as e:
        print("[ROUTER][WARN]", e)

    return keyword_fallback_domain(query)

# ----------------------------------------------------------------------
# TRANSLATION FOR RETRIEVAL (KB is EN)
# ----------------------------------------------------------------------

def translate_for_search(query: str, source_lang: str, target_lang: str, domain: str) -> str:
    source_lang = normalize_lang(source_lang)
    target_lang = normalize_lang(target_lang)

    if source_lang == target_lang:
        return query

    # pokud hledáme v EN KB, přeložíme do en
    sys_prompt = (
        "You translate helpdesk queries for semantic search.\n"
        "Rules:\n"
        "- Preserve technical meaning and entities\n"
        "- Keep it short\n"
        "- Do not answer the question\n"
        "- Return only the translated query text\n"
    )
    user_prompt = (
        f"Domain: {domain}\n"
        f"Translate from {source_lang} to {target_lang}:\n"
        f"{query}"
    )

    try:
        translated = post_chat(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.0,
            max_tokens=120,
            top_p=1.0,
            timeout_sec=20
        )
        translated = translated.strip().strip('"')
        if translated:
            print(f"[TRANSLATE][{domain}] {query} -> {translated}")
            return translated
    except Exception as e:
        print("[TRANSLATE][WARN]", e)

    return query

def force_answer_language(text: str, target_lang: str, domain: str) -> str:
    """
    Tvrdě vynutí jazyk odpovědi. Zachová číslované kroky.
    """
    target_lang = normalize_lang(target_lang)

    lang_names = {
        "cs": "Czech",
        "en": "English",
        "de": "German",
        "pl": "Polish",
        "it": "Italian",
    }
    target_name = lang_names.get(target_lang, "English")

    sys_prompt = (
        "You translate internal helpdesk answers.\n"
        "Rules:\n"
        "- Translate to the target language exactly\n"
        "- Preserve numbering (1., 2., 3., ...)\n"
        "- Preserve technical meaning\n"
        "- Keep commands, file paths, hostnames, service names unchanged\n"
        "- Do NOT add explanations\n"
        "- Return only the translated numbered steps"
    )

    user_prompt = (
        f"Target language: {target_name}\n"
        f"Domain: {domain}\n\n"
        f"Text to translate:\n{text}"
    )

    translated = post_chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=0.0,
        max_tokens=700,
        top_p=1.0,
        timeout_sec=30
    )

    return normalize_numbered_steps(translated)

# ----------------------------------------------------------------------
# VECTOR SEARCH
# ----------------------------------------------------------------------

def distance_label(dist: float, lang: str = "en") -> str:
    lang = normalize_lang(lang)
    labels = {
        "en": {
            "almost_same": "almost the same problem",
            "very_similar": "very similar incident",
            "related": "generally related topic",
            "noise": "likely noise / weak match",
        },
        "cs": {
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

def search_similar(domain: str, query_for_search: str, k: int = 5):
    if domain not in KB_RUNTIME:
        raise RuntimeError(f"KB domain not loaded: {domain}")

    kb = KB_RUNTIME[domain]
    index = kb["index"]
    metadata = kb["metadata"]

    vec = embed(query_for_search)
    distances, indices = index.search(vec, k)

    print(f"[SEARCH][{domain}] RAW D:", distances[0])
    print(f"[SEARCH][{domain}] RAW I:", indices[0])

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = metadata[int(idx)]
        results.append({
            "id": int(row.get("id", idx + 1)),
            "distance": float(dist),
            "problem": row.get("problem", ""),
            "symptoms": row.get("symptoms", ""),
            "analysis": row.get("analysis", ""),
            "solution": row.get("solution", ""),
            "domain": domain,
        })
    return results

def filter_similar(similar: List[dict]) -> List[dict]:
    similar = sorted(similar, key=lambda x: x["distance"])
    return [c for c in similar if c["distance"] <= SIMILAR_MAX_DIST]

def deduplicate(results: List[dict], limit: int = 3) -> List[dict]:
    seen = set()
    unique = []
    for r in results:
        key = (r.get("problem", "") or "").strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(r)
        if len(unique) >= limit:
            break
    return unique

def enrich_similar_cases(similar_cases: List[dict], reply_lang: str) -> List[dict]:
    for i, x in enumerate(similar_cases, start=1):
        x["distance_label"] = distance_label(x["distance"], reply_lang)
        x["case_id"] = x["id"]
        x["label"] = x["distance_label"]
        x["rank"] = i
    return similar_cases

def get_similar_cases(domain: str, query_for_search: str, reply_lang: str, k: int, dedup_limit: int = 3):
    raw = search_similar(domain, query_for_search, k=k)
    raw = filter_similar(raw)
    raw = deduplicate(raw, limit=dedup_limit)
    raw = enrich_similar_cases(raw, reply_lang)
    return raw

def format_similar_lines(similar_cases: List[dict]) -> List[str]:
    return [
        f"ID {x['id']} ({x['distance']:.3f}, {x['distance_label']}): {x['problem']}"
        for x in similar_cases
    ]

def build_context_text(similar_cases: List[dict]) -> str:
    if not similar_cases:
        return "No similar incidents found."

    parts = []
    for x in similar_cases:
        parts.append(
            f"- Similar incident [ID {x['id']}] ({x['domain']}, dist {x['distance']:.3f}, {x['distance_label']}):\n"
            f"  Problem: {x['problem']}\n"
            f"  Symptoms: {x['symptoms']}\n"
            f"  Analysis: {x['analysis']}\n"
            f"  Resolution: {x['solution']}"
        )
    return "\n\n".join(parts)

# ----------------------------------------------------------------------
# PROMPTS / LLM ANSWER
# ----------------------------------------------------------------------

def lang_instruction_text(lang: str) -> str:
    lang = normalize_lang(lang)
    m = {
        "cs": "Odpověz česky. Krátce, prakticky, krokově.",
        "en": "Answer in English. Short, practical, step-by-step.",
        "de": "Antworte auf Deutsch. Kurz, praktisch, schrittweise.",
        "pl": "Odpowiedz po polsku. Krótko, praktycznie, krok po kroku.",
        "it": "Rispondi in italiano. Breve, pratico, per passi.",
    }
    return m.get(lang, m["en"])

def get_system_prompt_for_domain(domain: str, reply_lang: str) -> str:
    lang_instruction = lang_instruction_text(reply_lang)

    key = f"LLM_SYS_PROMPT_{domain.upper()}"
    sys_tmpl = (os.getenv(key, "") or "").strip()

    if not sys_tmpl:
        sys_tmpl = (os.getenv("LLM_SYS_PROMPT", "") or "").strip()

    if not sys_tmpl:
        # hard fallback
        sys_tmpl = (
            "You are an internal service desk assistant for company operations.\n\n"
            "Your task is to produce short, actionable steps for the assigned domain.\n\n"
            "STRICT OUTPUT RULES:\n"
            "- Output MUST contain numbered steps only (1., 2., ...)\n"
            "- Each step on its own line\n"
            "- No headers or summaries\n"
            "- Keep steps practical\n"
            "{lang_instruction}"
        )

    sys_tmpl = sys_tmpl.replace("\\n", "\n")
    if "{lang_instruction}" in sys_tmpl:
        return sys_tmpl.format(lang_instruction=lang_instruction).strip()
    return (sys_tmpl + "\n" + lang_instruction).strip()

def ask_llm(query: str, context: str, domain: str, reply_lang: str = "cs"):
    reply_lang = normalize_lang(reply_lang)

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "400"))
    top_p = float(os.getenv("LLM_TOP_P", "1.0"))
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    sys_prompt = get_system_prompt_for_domain(domain, reply_lang)

    user_tmpl = (os.getenv("LLM_USER_PROMPT", "") or "").strip()
    if not user_tmpl:
        user_tmpl = (
            "Domain: {domain}\n\n"
            "User problem:\n{query}\n\n"
            "Similar historical incidents (retrieval context):\n{context}\n\n"
            "Rules:\n"
            "- Use the retrieval context if relevant\n"
            "- If context is weak/noisy, still give practical steps\n"
            "- Keep it short\n"
            "- Numbered steps only"
        )
    user_tmpl = user_tmpl.replace("\\n", "\n")

    user_prompt = user_tmpl.format(domain=domain, query=query, context=context).strip()

    start = time.time()
    raw_answer = post_chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        timeout_sec=timeout_sec,
    )

    fixed_answer = normalize_numbered_steps(raw_answer)

    # TVRDÉ vynucení jazyka odpovědi (model občas ignoruje lang instrukci)
    final_answer = force_answer_language(fixed_answer, reply_lang, domain)

    return {
        "answer": final_answer,
        "raw_answer": raw_answer,
        "response_time_seconds": round(time.time() - start, 2)
    }
    

# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------

def run_pipeline(query: str, request_lang: str, forced_domain: Optional[str] = None, k: int = 5, dedup_limit: int = 3):
    request_lang = normalize_lang(request_lang)

    # 1) domain routing
    if forced_domain and forced_domain in KB_RUNTIME:
        domain = forced_domain
        routing_reason = "forced_domain"
    else:
        domain, routing_reason = detect_domain_ai(query)
        if domain not in KB_RUNTIME:
            domain = DEFAULT_DOMAIN
            routing_reason += " -> fallback_unloaded_domain"

    # 2) retrieval language (KBs are EN in your setup)
    search_lang = normalize_lang(KB_CONFIG.get(domain, {}).get("search_lang", DEFAULT_SEARCH_LANG))

    # 3) translate query for retrieval if needed
    query_for_search = translate_for_search(query, request_lang, search_lang, domain) if search_lang != request_lang else query

    # 4) retrieval
    similar = get_similar_cases(domain, query_for_search, request_lang, k=k, dedup_limit=dedup_limit)
    context_text = build_context_text(similar)

    # 5) answer in request language
    llm_result = ask_llm(query=query, context=context_text, domain=domain, reply_lang=request_lang)

    return {
        "domain": domain,
        "routing_reason": routing_reason,
        "search_lang": search_lang,
        "query_for_search": query_for_search,
        "similar": similar,
        "llm_result": llm_result,
    }

# ----------------------------------------------------------------------
# ZAMMAD
# ----------------------------------------------------------------------

def zammad_headers():
    return {
        "Authorization": f"Token token={ZAMMAD_TOKEN}",
        "Content-Type": "application/json"
    }

def zammad_post_internal_note(ticket_id: int, text: str):
    if not ZAMMAD_URL or not ZAMMAD_TOKEN:
        print("[ZAMMAD][WARN] ZAMMAD_URL or ZAMMAD_TOKEN missing, skipping write-back")
        return

    url = f"{ZAMMAD_URL}/api/v1/ticket_articles"
    payload = {
        "ticket_id": ticket_id,
        "body": text,
        "type": "note",
        "internal": True
    }

    r = requests.post(url, headers=zammad_headers(), json=payload, timeout=30)
    print("=== ZAMMAD WRITE BACK ===")
    print("Status:", r.status_code)
    print("Response:", r.text[:2000])

def format_zammad_note(llm_answer: str, similar_cases: List[dict], domain: str, reply_lang: str) -> str:
    if normalize_lang(reply_lang) == "cs":
        head = f"AI návrh ({domain.upper()}):"
        sim_head = "Použité podobné incidenty:"
    else:
        head = f"AI suggestion ({domain.upper()}):"
        sim_head = "Similar incidents used:"

    lines = [head]
    if similar_cases:
        lines.append("")
        lines.append(sim_head)
        for line in format_similar_lines(similar_cases):
            lines.append(f"- {line}")

    lines.append("")
    lines.append(llm_answer.strip())
    return "\n".join(lines).strip()

# dedup webhook
PROCESSED = {}
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

def zammad_group_to_domain(group_name: str) -> Optional[str]:
    if not group_name:
        return None
    return GROUP_NAME_TO_DOMAIN.get(group_name.strip().lower())

# ----------------------------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_domains": sorted(list(KB_RUNTIME.keys())),
        "model": LLM_MODEL
    })

@app.route("/search", methods=["POST"])
def handle_search():
    start_total = time.time()
    data = request.json or {}

    query = (data.get("query") or "").strip()
    lang = normalize_lang(data.get("lang") or DEFAULT_REPLY_LANG)
    forced_domain = (data.get("domain") or "").strip().lower() or None

    if not query:
        return jsonify({"status": "error", "error": "missing query"}), 400

    p = run_pipeline(query, request_lang=lang, forced_domain=forced_domain, k=SIMILAR_K_SOLVE, dedup_limit=3)
    similar = p["similar"]

    log_interaction({
        "mode": "search",
        "query": query,
        "lang": lang,
        "domain": p["domain"],
        "routing_reason": p["routing_reason"],
        "search_lang": p["search_lang"],
        "query_for_search": p["query_for_search"],
        "similar_cases": [
            {"id": x["id"], "problem": x["problem"], "distance": x["distance"], "label": x["distance_label"], "domain": x["domain"]}
            for x in similar
        ],
        "total_time": round(time.time() - start_total, 2),
        "model": LLM_MODEL
    })

    return jsonify({
        "domain": p["domain"],
        "routing_reason": p["routing_reason"],
        "search_lang": p["search_lang"],
        "query_for_search": p["query_for_search"],
        "similar_cases": similar,
        "similar_cases_lines": format_similar_lines(similar),
        "similar_cases_text": "\n".join(format_similar_lines(similar)) if similar else ""
    })

@app.route("/solve", methods=["POST"])
def handle_solve():
    start_total = time.time()
    data = request.json or {}

    query = (data.get("query") or "").strip()
    lang = normalize_lang(data.get("lang") or DEFAULT_REPLY_LANG)
    forced_domain = (data.get("domain") or "").strip().lower() or None

    if not query:
        return jsonify({"status": "error", "error": "missing query"}), 400

    p = run_pipeline(query, request_lang=lang, forced_domain=forced_domain, k=SIMILAR_K_SOLVE, dedup_limit=3)

    llm_result = p["llm_result"]
    similar = p["similar"]
    total_time = round(time.time() - start_total, 2)

    similar_lines = format_similar_lines(similar)
    similar_text = "\n".join(similar_lines) if similar_lines else ""

    log_interaction({
        "mode": "solve",
        "query": query,
        "lang": lang,
        "domain": p["domain"],
        "routing_reason": p["routing_reason"],
        "search_lang": p["search_lang"],
        "query_for_search": p["query_for_search"],
        "similar_cases": [
            {"id": x["id"], "problem": x["problem"], "distance": x["distance"], "label": x["distance_label"], "domain": x["domain"]}
            for x in similar
        ],
        "answer": llm_result["answer"],
        "llm_time": llm_result["response_time_seconds"],
        "total_time": total_time,
        "model": LLM_MODEL
    })

    return jsonify({
        "domain": p["domain"],
        "routing_reason": p["routing_reason"],
        "search_lang": p["search_lang"],
        "reply_lang": lang,
        "llm_answer": llm_result["answer"],
        "similar_cases": similar,
        "similar_cases_lines": similar_lines,
        "similar_cases_text": similar_text,
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
        group = ticket.get("group", {}) or {}

        ticket_id = ticket.get("id")
        article_id = article.get("id")

        if not ticket_id:
            return jsonify({"status": "error", "error": "missing ticket.id"}), 400

        dedup_key = f"{ticket_id}:{article_id}" if article_id else f"{ticket_id}:{payload.get('event_id', 'no_event')}"
        if seen_recently(dedup_key):
            return jsonify({"status": "ok", "dedup": True})

        group_name = (group.get("name") or "").strip()
        group_name_l = group_name.lower()

        # jen povolené skupiny
        if AI_ENABLED_GROUPS and group_name_l not in AI_ENABLED_GROUPS:
            return jsonify({"status": "ok", "skipped": "group_not_enabled", "group": group_name})

        # nepřekrmovat sám sebe: ignoruj interní note od AI agenta/customer pokud je sender agent a interní note
        sender_name = ((article.get("sender") or {}) if isinstance(article.get("sender"), dict) else {})
        sender_type = (sender_name.get("name") or article.get("sender") or "").lower() if sender_name else str(article.get("sender", "")).lower()
        is_internal_note = bool(article.get("internal"))

        # pokud webhook přiletěl na interní poznámku od agenta, typicky nechceme znovu zpracovat
        if is_internal_note:
            return jsonify({"status": "ok", "skipped": "internal_note"})

        forced_domain = zammad_group_to_domain(group_name)

        title = (ticket.get("title") or "").strip()
        body = (article.get("body") or "").strip()
        query = f"{title}\n\n{body}".strip()

        # detekce jazyka z payloadu (volitelně), jinak default
        # Tady můžeš později přidat AI language detection; zatím držíme default cs.
        reply_lang = DEFAULT_REPLY_LANG

        p = run_pipeline(query, request_lang=reply_lang, forced_domain=forced_domain, k=SIMILAR_K_ZAMMAD, dedup_limit=3)

        answer_note = format_zammad_note(
            llm_answer=p["llm_result"]["answer"],
            similar_cases=p["similar"],
            domain=p["domain"],
            reply_lang=reply_lang
        )

        print("=== QUERY SENT TO LLM ===")
        print(query[:1000])
        print("=== LLM ANSWER ===")
        print(answer_note[:4000])

        zammad_post_internal_note(ticket_id, answer_note)

        log_interaction({
            "mode": "zammad",
            "ticket_id": ticket_id,
            "article_id": article_id,
            "group": group_name,
            "forced_domain": forced_domain,
            "domain": p["domain"],
            "routing_reason": p["routing_reason"],
            "reply_lang": reply_lang,
            "search_lang": p["search_lang"],
            "query_preview": query[:500],
            "query_for_search": p["query_for_search"],
            "similar_cases": [
                {"id": x["id"], "problem": x["problem"], "distance": round(x["distance"], 4), "label": x["distance_label"], "domain": x["domain"]}
                for x in p["similar"]
            ],
            "answer": answer_note,
            "llm_time": p["llm_result"]["response_time_seconds"],
            "total_time": round(p["llm_result"]["response_time_seconds"], 2),
            "model": LLM_MODEL
        })

        return jsonify({"status": "ok", "domain": p["domain"]})

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