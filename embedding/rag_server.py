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
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1").rstrip("/")
EMBED_URL = os.getenv("EMBED_URL", f"{LMSTUDIO_BASE_URL}/embeddings")
CHAT_URL = os.getenv("CHAT_URL", f"{LMSTUDIO_BASE_URL}/chat/completions")

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen/qwen3-4b-2507")

# sjednocení: podporujeme cs i cz, interně normalizujeme na "cs"
ALLOWED_LANGS = ["en", "cs", "cz", "de", "pl", "it"]

SIMILAR_MAX_DIST = float(os.getenv("SIMILAR_MAX_DIST", "0.84"))
SIMILAR_K_SOLVE = int(os.getenv("SIMILAR_K_SOLVE", "5"))
SIMILAR_K_ZAMMAD = SIMILAR_K_SOLVE

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

# ----------------------------------------------------------------------
# ZAMMAD CONFIG
# ----------------------------------------------------------------------

ZAMMAD_URL = os.getenv("ZAMMAD_URL", "http://127.0.0.1:8080").rstrip("/")
ZAMMAD_TOKEN = os.getenv("ZAMMAD_TOKEN")

if not ZAMMAD_TOKEN:
    raise RuntimeError("ZAMMAD_TOKEN is not set (check .env or environment variables)")

INTERNAL_EMAIL_DOMAIN = (os.getenv("INTERNAL_EMAIL_DOMAIN", "ventureout.cz") or "").strip().lower()

# Skupiny, ve kterých má AI běžet
AI_ENABLED_GROUPS = {
    x.strip().lower()
    for x in (os.getenv("AI_ENABLED_GROUPS", "IT,HR,Finance,Onboarding") or "").split(",")
    if x.strip()
}

DEFAULT_DOMAIN = (os.getenv("DEFAULT_DOMAIN", "it") or "it").strip().lower()
DEFAULT_REPLY_LANG = (os.getenv("DEFAULT_REPLY_LANG", "cs") or "cs").strip().lower()
DEFAULT_SEARCH_LANG = (os.getenv("DEFAULT_SEARCH_LANG", "cs") or "cs").strip().lower()

# mapování názvu skupiny v Zammadu -> doména
GROUP_TO_DOMAIN = {
    (os.getenv("GROUP_MAP_IT", "IT") or "IT").strip().lower(): "it",
    (os.getenv("GROUP_MAP_HR", "HR") or "HR").strip().lower(): "hr",
    (os.getenv("GROUP_MAP_FINANCE", "Finance") or "Finance").strip().lower(): "finance",
    (os.getenv("GROUP_MAP_ONBOARDING", "Onboarding") or "Onboarding").strip().lower(): "onboarding",
}

SEARCH_LANG_BY_DOMAIN = {
    "it": (os.getenv("IT_SEARCH_LANG", "en") or "en").strip().lower(),
    "hr": (os.getenv("HR_SEARCH_LANG", "cs") or "cs").strip().lower(),
    "finance": (os.getenv("FINANCE_SEARCH_LANG", "cs") or "cs").strip().lower(),
    "onboarding": (os.getenv("ONBOARDING_SEARCH_LANG", "cs") or "cs").strip().lower(),
}

DOMAIN_PROMPT_ENV = {
    "it": "LLM_SYS_PROMPT_IT",
    "hr": "LLM_SYS_PROMPT_HR",
    "finance": "LLM_SYS_PROMPT_FINANCE",
    "onboarding": "LLM_SYS_PROMPT_ONBOARDING",
}

# ----------------------------------------------------------------------
# KB CONFIG (4 domény)
# ----------------------------------------------------------------------

def _abs_path_from_env(key: str, fallback_rel: str) -> str:
    raw = (os.getenv(key, fallback_rel) or fallback_rel).strip()
    if os.path.isabs(raw):
        return raw
    return os.path.abspath(os.path.join(ROOT_DIR, raw))

KB_CONFIG = {
    "it": {
        "jsonl": _abs_path_from_env("KB_IT_JSONL", "data/it/problems.jsonl"),
        "index": _abs_path_from_env("KB_IT_INDEX", "vectordb/it/faiss.index"),
        "meta": _abs_path_from_env("KB_IT_META", "vectordb/it/meta.pkl"),
    },
    "hr": {
        "jsonl": _abs_path_from_env("KB_HR_JSONL", "data/hr/problems.jsonl"),
        "index": _abs_path_from_env("KB_HR_INDEX", "vectordb/hr/faiss.index"),
        "meta": _abs_path_from_env("KB_HR_META", "vectordb/hr/meta.pkl"),
    },
    "finance": {
        "jsonl": _abs_path_from_env("KB_FINANCE_JSONL", "data/finance/problems.jsonl"),
        "index": _abs_path_from_env("KB_FINANCE_INDEX", "vectordb/finance/faiss.index"),
        "meta": _abs_path_from_env("KB_FINANCE_META", "vectordb/finance/meta.pkl"),
    },
    "onboarding": {
        "jsonl": _abs_path_from_env("KB_ONBOARDING_JSONL", "data/onboarding/problems.jsonl"),
        "index": _abs_path_from_env("KB_ONBOARDING_INDEX", "vectordb/onboarding/faiss.index"),
        "meta": _abs_path_from_env("KB_ONBOARDING_META", "vectordb/onboarding/meta.pkl"),
    },
}

# ----------------------------------------------------------------------
# LOGGING
# ----------------------------------------------------------------------

LOG_DIR = os.path.join(BASE_DIR, "logs")
LOG_FILE = os.path.join(LOG_DIR, "rag.log.jsonl")

def log_interaction(payload: dict):
    os.makedirs(LOG_DIR, exist_ok=True)
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

# ----------------------------------------------------------------------
# LOAD ALL VECTOR DBs
# ----------------------------------------------------------------------

KB_RUNTIME = {}  # domain -> {"index":..., "metadata":[...], ...}

def load_kb(domain: str, cfg: dict):
    idx_path = cfg["index"]
    meta_path = cfg["meta"]
    if not os.path.exists(idx_path):
        raise FileNotFoundError(f"Missing FAISS index for '{domain}': {idx_path}")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.pkl for '{domain}': {meta_path}")

    index = faiss.read_index(idx_path)
    with open(meta_path, "rb") as f:
        metadata = pickle.load(f)

    KB_RUNTIME[domain] = {
        "index": index,
        "metadata": metadata,
        "index_path": idx_path,
        "meta_path": meta_path,
    }

for _domain, _cfg in KB_CONFIG.items():
    try:
        load_kb(_domain, _cfg)
        print(f"[KB] Loaded {_domain}: {KB_RUNTIME[_domain]['index_path']}")
    except Exception as e:
        print(f"[KB] WARNING: {_domain} not loaded: {e}")

if DEFAULT_DOMAIN not in KB_RUNTIME and KB_RUNTIME:
    DEFAULT_DOMAIN = list(KB_RUNTIME.keys())[0]

if not KB_RUNTIME:
    raise RuntimeError("No KB loaded. Check KB_* paths in .env and FAISS files.")

# ----------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------

def normalize_lang(lang: str) -> str:
    x = (lang or "").strip().lower()
    if x == "cz":
        return "cs"
    if x not in ["en", "cs", "de", "pl", "it"]:
        return "en"
    return x

def normalize_numbered_steps(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    steps = []
    for l in lines:
        m = re.match(r"^\s*\d+\.\s*(.*)$", l)
        if not m:
            continue
        body = m.group(1).strip()
        if body.startswith("$") and steps:
            steps[-1] = steps[-1].rstrip() + " " + body
        else:
            steps.append(body)

    if not steps and text.strip():
        # fallback: když model nedodrží formát
        compact = " ".join([x.strip() for x in text.splitlines() if x.strip()])
        return f"1. {compact}" if compact else ""

    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, start=1))

def filter_similar(similar):
    similar = sorted(similar, key=lambda x: x["distance"])
    return [c for c in similar if c["distance"] <= SIMILAR_MAX_DIST]

def embed(text: str) -> np.ndarray:
    r = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    )

    try:
        j = r.json()
    except Exception:
        raise RuntimeError(f"Embedding API returned non-JSON. status={r.status_code} body={r.text[:500]}")

    if r.status_code >= 400:
        raise RuntimeError(f"Embedding API error. status={r.status_code} body={j}")

    if "data" not in j or not j["data"]:
        raise RuntimeError(f"Embedding API bad response: {j}")

    return np.array(j["data"][0]["embedding"], dtype="float32").reshape(1, -1)

def search_similar(query: str, domain: str, k: int = 5):
    if domain not in KB_RUNTIME:
        raise RuntimeError(f"Domain '{domain}' KB is not loaded.")

    vec = embed(query)
    index = KB_RUNTIME[domain]["index"]
    metadata = KB_RUNTIME[domain]["metadata"]

    distances, indices = index.search(vec, k)

    print(f"[SEARCH][{domain}] RAW D:", distances[0])
    print(f"[SEARCH][{domain}] RAW I:", indices[0])

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0:
            continue
        row = metadata[int(idx)]
        results.append({
            "id": int(row.get("id", idx)),
            "distance": float(dist),
            "problem": row.get("problem", ""),
            "symptoms": row.get("symptoms", ""),
            "analysis": row.get("analysis", ""),
            "solution": row.get("solution", ""),
            "domain": domain,
        })
    return results

def deduplicate(results, limit: int = 3):
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

def distance_label(dist: float, lang: str = "en") -> str:
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

    d = labels.get(normalize_lang(lang), labels["en"])

    if dist < 0.80:
        return d["almost_same"]
    if dist < 0.88:
        return d["very_similar"]
    if dist < 0.95:
        return d["related"]
    return d["noise"]

def enrich_similar_cases(similar_cases, lang: str):
    lang = normalize_lang(lang)
    for i, x in enumerate(similar_cases, start=1):
        x["distance_label"] = distance_label(x["distance"], lang)
        x["case_id"] = x["id"]
        x["label"] = x["distance_label"]
        x["rank"] = i
    return similar_cases

def get_similar_cases(query: str, domain: str, label_lang: str, k: int, dedup_limit: int = 3):
    raw = search_similar(query, domain=domain, k=k)
    raw = filter_similar(raw)
    raw = deduplicate(raw, limit=dedup_limit)
    raw = enrich_similar_cases(raw, label_lang)
    return raw

def build_context_text(similar_cases) -> str:
    parts = []
    for x in similar_cases:
        parts.append(
            f"- Similar incident [ID {x['id']}] (dist {x['distance']:.3f}, {x['distance_label']}):\n"
            f"  Problem: {x['problem']}\n"
            f"  Symptoms: {x.get('symptoms','')}\n"
            f"  Resolution: {x.get('solution','')}"
        )
    return "\n\n".join(parts)

def format_similar_lines(similar_cases) -> list[str]:
    return [
        f"ID {x['id']} ({x['distance']:.3f}, {x['distance_label']}): {x['problem']}"
        for x in similar_cases
    ]

def get_lang_instruction(lang: str) -> str:
    lang = normalize_lang(lang)
    lang_map = {
        "en": "Write answer in English. Short, technical, operational.",
        "cs": "Odpověz česky. Stručně, technicky, provozní kroky.",
        "de": "Antwort auf Deutsch. Kurz und technisch.",
        "pl": "Odpowiedz po polsku. Krótko i technicznie.",
        "it": "Rispondi in italiano. Breve e tecnico."
    }
    return lang_map.get(lang, lang_map["en"])

def get_system_prompt_for_domain(domain: str, reply_lang: str) -> str:
    lang_instruction = get_lang_instruction(reply_lang)

    env_key = DOMAIN_PROMPT_ENV.get(domain, "")
    sys_tmpl = (os.getenv(env_key, "") or "").strip() if env_key else ""

    if not sys_tmpl:
        sys_tmpl = (os.getenv("LLM_SYS_PROMPT", "") or "").strip()

    if not sys_tmpl:
        sys_tmpl = (
            "You are an internal service desk assistant.\n"
            "Return numbered operational steps only.\n"
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
            "User problem:\n{query}\n\n"
            "Similar historical incidents (if any):\n{context}\n\n"
            "Provide recommended fix steps."
        )
    user_tmpl = user_tmpl.replace("\\n", "\n")

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

    raw_answer = j["choices"][0]["message"]["content"] or ""
    fixed_answer = normalize_numbered_steps(raw_answer)

    return {
        "answer": fixed_answer,
        "raw_answer": raw_answer,
        "response_time_seconds": round(time.time() - start, 2)
    }

def translate_query_for_retrieval(query: str, target_lang: str = "en") -> str:
    """
    Krátký technický překlad dotazu pro retrieval.
    Překládá jen dotaz, ne odpověď.
    """
    target_lang = normalize_lang(target_lang)

    # pokud je target EN, přelož do EN; jinak můžeš vracet původní
    # (teď máš KB anglicky, takže target bude hlavně en)
    if target_lang != "en":
        return query

    timeout_sec = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    sys_prompt = (
        "You are a translation helper for semantic search.\n"
        "Translate the user query to concise English for retrieval.\n"
        "Keep technical terms, product names, usernames, hostnames, and error messages unchanged.\n"
        "Return only the translated text, no commentary."
    )

    r = requests.post(
        CHAT_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": query}
            ],
            "temperature": 0.0,
            "max_tokens": 200
        },
        timeout=timeout_sec
    )

    try:
        j = r.json()
    except Exception:
        return query  # fallback: nepoložit pipeline

    if r.status_code >= 400 or "choices" not in j or not j["choices"]:
        return query

    out = (j["choices"][0]["message"]["content"] or "").strip()
    return out if out else query

def detect_domain_ai(query: str, zammad_group_name: str | None = None) -> tuple[str, str]:
    """
    AI router:
    - Pokud přijde známá Zammad skupina, použije mapování (rychlé a spolehlivé)
    - Jinak zavolá LLM a nechá ho vrátit jednu doménu: it/hr/finance/onboarding
    - Když LLM selže, fallback na DEFAULT_DOMAIN
    """
    # 1) group override (když webhook nese group name, je to nejlepší signál)
    group_name = (zammad_group_name or "").strip()
    group_map = {
        (os.getenv("GROUP_MAP_IT", "IT") or "").strip().lower(): "it",
        (os.getenv("GROUP_MAP_HR", "HR") or "").strip().lower(): "hr",
        (os.getenv("GROUP_MAP_FINANCE", "Finance") or "").strip().lower(): "finance",
        (os.getenv("GROUP_MAP_ONBOARDING", "Onboarding") or "").strip().lower(): "onboarding",
    }

    if group_name:
        g = group_name.lower()
        if g in group_map:
            return group_map[g], f"group_map:{group_name}"

    # 2) AI klasifikátor přes LLM
    timeout_sec = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))

    system_prompt = (
        "You are a domain classifier for an internal helpdesk.\n"
        "Classify the user request into exactly one domain from this set:\n"
        "- it\n"
        "- hr\n"
        "- finance\n"
        "- onboarding\n\n"
        "Rules:\n"
        "- Return ONLY valid JSON, no markdown, no explanation.\n"
        "- JSON format: {\"domain\":\"it|hr|finance|onboarding\",\"confidence\":0.0,\"reason\":\"short\"}\n"
        "- 'confidence' is 0..1\n"
        "- 'reason' is very short English phrase\n"
        "- If unsure, choose the closest operational domain.\n"
    )

    user_prompt = f"Request:\n{(query or '').strip()}"

    try:
        r = requests.post(
            CHAT_URL,
            json={
                "model": LLM_MODEL,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": 80,
            },
            timeout=timeout_sec
        )

        j = r.json()
        if r.status_code >= 400:
            raise RuntimeError(f"Classifier API error: {j}")

        raw = (j["choices"][0]["message"]["content"] or "").strip()

        # zkus čistý JSON
        try:
            parsed = json.loads(raw)
        except Exception:
            # občas model vrátí text kolem -> zkus vytáhnout první {...}
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise RuntimeError(f"Classifier returned non-JSON: {raw[:300]}")
            parsed = json.loads(m.group(0))

        domain = str(parsed.get("domain", "")).strip().lower()
        conf = parsed.get("confidence", None)
        reason = str(parsed.get("reason", "")).strip()

        if domain not in ("it", "hr", "finance", "onboarding"):
            raise RuntimeError(f"Invalid domain from classifier: {domain}")

        return domain, f"ai_classifier:{domain}:{conf}:{reason}"

    except Exception as e:
        fallback = (os.getenv("DEFAULT_DOMAIN", "it") or "it").strip().lower()
        if fallback not in ("it", "hr", "finance", "onboarding"):
            fallback = "it"
        return fallback, f"fallback:{fallback}:classifier_error:{str(e)[:120]}"

def normalize_lang(lang: str) -> str:
    x = (lang or "").strip().lower()
    aliases = {
        "cz": "cs",
        "cs": "cs",
        "en": "en",
        "de": "de",
        "pl": "pl",
        "it": "it",
    }
    return aliases.get(x, "cs")


def get_search_lang_for_domain(domain: str) -> str:
    domain = (domain or "it").strip().lower()
    env_key = {
        "it": "IT_SEARCH_LANG",
        "hr": "HR_SEARCH_LANG",
        "finance": "FINANCE_SEARCH_LANG",
        "onboarding": "ONBOARDING_SEARCH_LANG",
    }.get(domain, "DEFAULT_SEARCH_LANG")

    return normalize_lang(os.getenv(env_key, os.getenv("DEFAULT_SEARCH_LANG", "en")))

def get_system_prompt_for_domain(domain: str, reply_lang: str) -> str:
    domain = (domain or "it").strip().lower()
    reply_lang = normalize_lang(reply_lang)

    lang_map = {
        "cs": "Odpověz česky.",
        "en": "Write the answer in English.",
        "de": "Antwort auf Deutsch.",
        "pl": "Odpowiedz po polsku.",
        "it": "Rispondi in italiano.",
    }
    lang_instruction = lang_map.get(reply_lang, "Odpověz česky.")

    env_key = {
        "it": "LLM_SYS_PROMPT_IT",
        "hr": "LLM_SYS_PROMPT_HR",
        "finance": "LLM_SYS_PROMPT_FINANCE",
        "onboarding": "LLM_SYS_PROMPT_ONBOARDING",
    }.get(domain, "LLM_SYS_PROMPT")

    sys_tmpl = (os.getenv(env_key, "") or "").strip() or (os.getenv("LLM_SYS_PROMPT", "") or "").strip()

    if not sys_tmpl:
        sys_tmpl = "You are an internal helpdesk assistant.\nOutput numbered steps only.\n{lang_instruction}"

    sys_tmpl = sys_tmpl.replace("\\n", "\n")
    if "{lang_instruction}" in sys_tmpl:
        return sys_tmpl.format(lang_instruction=lang_instruction)

    return sys_tmpl + "\n" + lang_instruction

# ----------------------------------------------------------------------
# ROUTING / CLASSIFICATION
# ----------------------------------------------------------------------

DOMAIN_KEYWORDS = {
    "it": [
        "server", "linux", "vpn", "mail", "email", "printer", "network", "ssh", "dns", "login",
        "heslo", "přihlás", "prihlas", "pošta", "mailbox", "wifi", "outlook", "notebook", "laptop",
        "access denied", "reset password", "active directory", "ldap"
    ],
    "hr": [
        "dovolen", "nemoc", "neschopen", "mateř", "mater", "otcov", "paragraf", "pracovní", "pracovni",
        "mzda", "výplata", "vyplata", "směna", "smena", "pracovní doba", "docházka", "dochazka",
        "hr", "zaměstnan", "zamestnan", "zákoník práce", "zakonik"
    ],
    "finance": [
        "faktura", "invoice", "platba", "payment", "účet", "ucet", "iban", "dph", "vat", "cashflow",
        "splatnost", "dodavatel", "supplier", "úhrada", "uhrada", "převod", "prevod", "finance",
        "rozpočet", "rozpocet", "vyúčtování", "vyuctovani"
    ],
    "onboarding": [
        "onboarding", "nováček", "novacek", "nástup", "nastup", "first day", "první den", "prvni den",
        "access", "přístup", "pristup", "účet", "ucet", "notebook", "equipment", "vybavení", "vybaveni",
        "konto", "úvodní školení", "uvodni skoleni"
    ],
}

def classify_domain_from_text(text: str) -> tuple[str, str]:
    """
    Jednoduchá heuristika. Když časem budeš chtít, dá se nahradit LLM klasifikací.
    Vrací (domain, reason)
    """
    t = (text or "").lower()
    scores = {d: 0 for d in DOMAIN_KEYWORDS.keys()}

    for domain, words in DOMAIN_KEYWORDS.items():
        for w in words:
            if w in t:
                scores[domain] += 1

    best_domain = max(scores, key=scores.get)
    best_score = scores[best_domain]

    if best_score <= 0:
        return DEFAULT_DOMAIN, "fallback_default_domain"

    return best_domain, f"keyword_classifier:{scores}"

def resolve_domain(ticket: dict | None, query: str, requested_domain: str | None = None) -> tuple[str, str]:
    # 1) explicitní doména z API requestu
    if requested_domain:
        d = requested_domain.strip().lower()
        if d in KB_RUNTIME:
            return d, "request.domain"

    # 2) Zammad group name
    if ticket:
        group = (ticket.get("group") or {}) if isinstance(ticket.get("group"), dict) else {}
        group_name = (group.get("name") or "").strip().lower()
        if group_name:
            mapped = GROUP_TO_DOMAIN.get(group_name)
            if mapped and mapped in KB_RUNTIME:
                return mapped, f"zammad.group:{group_name}"

    # 3) fallback klasifikace z textu
    d, reason = classify_domain_from_text(query)
    if d in KB_RUNTIME:
        return d, reason

    # 4) hard fallback
    return DEFAULT_DOMAIN, "hard_fallback"

def detect_reply_lang(payload_lang: str | None = None) -> str:
    if payload_lang:
        return normalize_lang(payload_lang)
    return normalize_lang(DEFAULT_REPLY_LANG)

def detect_search_lang(domain: str) -> str:
    lang = SEARCH_LANG_BY_DOMAIN.get(domain, DEFAULT_SEARCH_LANG)
    return normalize_lang(lang)

def maybe_translate_query_for_retrieval(query: str, domain: str, search_lang: str) -> str:
    search_lang = normalize_lang(search_lang)
    if search_lang == "en":
        translated = translate_query_for_retrieval(query, target_lang="en")
        print(f"[TRANSLATE][{domain}] {query[:120]} -> {translated[:120]}")
        return translated
    return query

# ----------------------------------------------------------------------
# PIPELINE
# ----------------------------------------------------------------------

def run_pipeline(query: str, domain: str, reply_lang: str, k: int, dedup_limit: int = 3):
    # retrieval language (např. IT=en)
    search_lang = detect_search_lang(domain)

    # tady časem může být LLM překlad query -> search_lang
    retrieval_query = maybe_translate_query_for_retrieval(query, domain, search_lang)

    similar = get_similar_cases(
        query=retrieval_query,
        domain=domain,
        label_lang=reply_lang,
        k=k,
        dedup_limit=dedup_limit
    )

    context = build_context_text(similar)
    llm_result = ask_llm(query=query, context=context, domain=domain, reply_lang=reply_lang)

    return llm_result, similar, {
        "domain": domain,
        "reply_lang": reply_lang,
        "search_lang": search_lang,
        "retrieval_query": retrieval_query,
    }

# ----------------------------------------------------------------------
# ZAMMAD WRITE BACK
# ----------------------------------------------------------------------

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

    r = requests.post(url, headers=headers, json=payload, timeout=30)

    print("=== ZAMMAD WRITE BACK ===")
    print("Status:", r.status_code)
    print("Response:", r.text)

    if r.status_code >= 400:
        raise RuntimeError(f"Zammad write-back failed: {r.status_code} {r.text[:500]}")

def format_zammad_note(llm_answer: str, similar_cases, domain: str, routing_reason: str) -> str:
    header_lines = [
        f"AI domain: {domain}",
        f"Routing: {routing_reason}",
    ]

    if similar_cases:
        header_lines.append("Similar historical incidents used:")
        header_lines.extend([f"- {line}" for line in format_similar_lines(similar_cases)])

    return "\n".join(header_lines) + "\n\n" + llm_answer.strip()

# ----------------------------------------------------------------------
# ZAMMAD WEBHOOK DEDUP
# ----------------------------------------------------------------------

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

# ----------------------------------------------------------------------
# API ENDPOINTS
# ----------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "loaded_domains": sorted(list(KB_RUNTIME.keys())),
        "default_domain": DEFAULT_DOMAIN,
        "model": LLM_MODEL,
        "embed_model": EMBED_MODEL,
    })

@app.route("/classify", methods=["POST"])
def handle_classify():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    domain, reason = resolve_domain(ticket=None, query=query, requested_domain=data.get("domain"))
    return jsonify({"domain": domain, "reason": reason})

@app.route("/search", methods=["POST"])
def handle_search():
    data = request.json or {}
    query = (data.get("query") or "").strip()
    req_lang = detect_reply_lang(data.get("lang"))

    domain, routing_reason = resolve_domain(ticket=None, query=query, requested_domain=data.get("domain"))

    results = get_similar_cases(
        query=maybe_translate_query_for_retrieval(query, domain, detect_search_lang(domain)),
        domain=domain,
        label_lang=req_lang,
        k=SIMILAR_K_SOLVE,
        dedup_limit=3
    )

    log_interaction({
        "mode": "search",
        "query": query,
        "domain": domain,
        "routing_reason": routing_reason,
        "reply_lang": req_lang,
        "search_lang": detect_search_lang(domain),
        "similar_cases": [
            {"id": x["id"], "problem": x["problem"], "distance": x["distance"], "label": x["distance_label"]}
            for x in results
        ],
        "model": LLM_MODEL
    })

    return jsonify({
        "domain": domain,
        "routing_reason": routing_reason,
        "search_lang": detect_search_lang(domain),
        "similar_cases": results
    })

@app.route("/solve", methods=["POST"])
def handle_solve():
    start_total = time.time()
    data = request.json or {}

    query = (data.get("query") or "").strip()
    reply_lang = detect_reply_lang(data.get("lang"))

    domain, routing_reason = resolve_domain(ticket=None, query=query, requested_domain=data.get("domain"))

    llm_result, similar, dbg = run_pipeline(
        query=query,
        domain=domain,
        reply_lang=reply_lang,
        k=SIMILAR_K_SOLVE,
        dedup_limit=3
    )

    total_time = round(time.time() - start_total, 2)
    similar_lines = format_similar_lines(similar) if similar else []
    similar_text = "\n".join(similar_lines) if similar_lines else ""

    log_interaction({
        "mode": "solve",
        "query": query,
        "domain": domain,
        "routing_reason": routing_reason,
        "reply_lang": reply_lang,
        "search_lang": dbg["search_lang"],
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
        "domain": domain,
        "routing_reason": routing_reason,
        "reply_lang": reply_lang,
        "search_lang": dbg["search_lang"],
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

        ticket_id = ticket.get("id")
        article_id = article.get("id")

        if not ticket_id:
            return jsonify({"status": "error", "error": "missing ticket.id"}), 400

        dedup_key = f"{ticket_id}:{article_id}" if article_id else f"{ticket_id}:{payload.get('event_id', 'no_event')}"
        if seen_recently(dedup_key):
            return jsonify({"status": "ok", "dedup": True})

        # filtr: jen první článek ticketu (aby se necyklilo)
        article_count = int(ticket.get("article_count") or 0)
        if article_count > 1:
            return jsonify({"status": "ok", "skipped": "not_first_article", "article_count": article_count})

        # filtr: jen povolené skupiny
        group_name = ((ticket.get("group") or {}).get("name") or "").strip()
        domain, routing_reason = detect_domain_ai(query, zammad_group_name=group_name)
        reply_lang = normalize_lang(os.getenv("DEFAULT_REPLY_LANG", "cs"))
        search_lang = get_search_lang_for_domain(domain)

        if group_name and AI_ENABLED_GROUPS:
            if group_name.lower() not in AI_ENABLED_GROUPS:
                return jsonify({"status": "ok", "skipped": "group_not_enabled", "group": group_name})

        # filtr: neodpovídat na interní AI poznámku (aby se nekrmilo samo)
        sender_name = (article.get("sender") or article.get("from") or "").lower()
        article_type = (article.get("type") or "").lower()
        if article_type == "note" and ("ai@" in sender_name or "assistant" in sender_name):
            return jsonify({"status": "ok", "skipped": "self_note"})

        title = (ticket.get("title") or "").strip()
        body = (article.get("body") or "").strip()
        query = f"{title}\n\n{body}".strip()

        domain, routing_reason = detect_domain_ai(query)
        reply_lang = normalize_lang((data.get("lang") or os.getenv("DEFAULT_REPLY_LANG", "cs")))
        search_lang = get_search_lang_for_domain(domain)
        search_query = maybe_translate_for_search(query, source_lang=reply_lang, target_lang=search_lang, domain=domain)

        print("=== ROUTING ===")
        print("domain:", domain, "| reason:", routing_reason)
        print("=== QUERY SENT TO PIPELINE ===")
        print(query)

        start_total = time.time()
        llm_result, similar, dbg = run_pipeline(
            query=query,
            domain=domain,
            reply_lang=reply_lang,
            k=SIMILAR_K_ZAMMAD,
            dedup_limit=3
        )
        total_time = round(time.time() - start_total, 2)

        answer_note = format_zammad_note(
            llm_answer=llm_result["answer"],
            similar_cases=similar,
            domain=domain,
            routing_reason=routing_reason
        )

        print("=== LLM ANSWER ===")
        print(answer_note)

        zammad_post_internal_note(ticket_id, answer_note)

        log_interaction({
            "mode": "zammad",
            "ticket_id": ticket_id,
            "article_id": article_id,
            "group_name": group_name,
            "domain": domain,
            "routing_reason": routing_reason,
            "reply_lang": reply_lang,
            "search_lang": dbg["search_lang"],
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

        return jsonify({"status": "ok", "domain": domain, "routing_reason": routing_reason})

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

