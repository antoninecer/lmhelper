LM Helper – IT Outsourcing AI

Inteligentní RAG systém pro rychlou diagnostiku IT problémů
(Linux, síťové technologie, firewalling, storage, virtualizace, Cisco, FortiGate, VMware…)

🚀 Co projekt dělá

LM Helper je lokální AI nástroj, který kombinuje:

RAG vyhledávání pomocí FAISS
→ Vyhledá nejpodobnější reálné IT problémy z vlastní znalostní báze.

Lokální jazykový model (Qwen2.5 nebo jiný z LM Studio)
→ Sestaví finální technickou odpověď, doporučení, nebo postup řešení.

Webový frontend
→ Jednoduchá webová aplikace (HTML + JS), která umožňuje zadat dotaz a okamžitě vidět výsledek.

Celý systém běží lokálně, bez cloudových služeb a bez úniku dat.

🧱 Architektura
[Frontend] → [Flask RAG server] → [FAISS] → [LM Studio / embeddings API]

🔹 Backend (Python / Flask)

přijímá dotazy z webu (/search)

vytvoří embedding z dotazu

FAISS najde 3 nejbližší existující problémy

data vrátí klientovi ve formátu JSON

🔹 Frontend (HTML/JS)

jednoduché UI pro zadávání dotazů

komunikuje přímo s Flask serverem

🔹 Vektorová databáze

620 IT problémů

každá položka: problem, symptoms, analysis, solution

embeddings: text-embedding-nomic-embed-text-v1.5

uložené soubory:

vectordb/faiss.index

vectordb/meta.pkl

📦 Instalace
1️⃣ Klonování repozitáře
git clone https://github.com/USERNAME/lmhelper.git
cd lmhelper

2️⃣ Vytvoření a aktivace virtualenv
python3 -m venv venv
source venv/bin/activate

3️⃣ Instalace závislostí
pip install -r requirements.txt

4️⃣ Spuštění LM Studio serveru

Načti:

text-embedding-nomic-embed-text-v1.5 (typ embedding)

Qwen2.5-7B-Instruct-MLX (nebo jiný model podle potřeby)

Zapni API na:

http://127.0.0.1:9999

5️⃣ Naplnění FAISS vektorové DB

Jen při prvním spuštění:

cd embedding
python3 embed_faiss.py

6️⃣ Spuštění RAG serveru
python3 rag_server.py


Server běží na:

http://127.0.0.1:5001/search

7️⃣ Otevření webového UI

V browseru:

http://127.0.0.1:8080/

🔍 Použití API
Request:
POST /search
{
  "query": "uživatel se nemůže připojit přes VPN"
}

Response:
[
  {
    "problem": "VPN client connects but no internet",
    "solution": "enable split tunneling",
    "analysis": "split-tunnel misconfigured",
    "symptoms": "default gateway overridden",
    "distance": 0.84
  }
]

📁 Struktura projektu
lmhelper/
│
├── embedding/
│   ├── embed_faiss.py
│   ├── search_faiss.py
│   └── rag_server.py
│
├── vectordb/
│   ├── faiss.index
│   └── meta.pkl
│
├── www/
│   ├── index.php
│   ├── call_llm.php   (volání LLM, pokud se použije)
│   └── style.css
│
├── data/
│   └── problems.jsonl
│
└── README.md

🔥 Budoucí rozšíření

přidat LLM reasoning: model vysvětlí, proč problém vznikl

přidat automatické generování odpovědí

rozšířit dataset na 2000+ problémů

přidat OAuth2 / API klíče

přidat "Learning mode": systém se učí z nových tiketů

🔌 Integrace se Zammad (Helpdesk)

LM Helper lze přímo napojit na Zammad Helpdesk a automaticky:

analyzovat nově vytvořené nebo otevřené tickety

vyhledat podobné historické incidenty ve vektorové databázi

vygenerovat návrh řešení pomocí lokálního LLM

vložit odpověď zpět do ticketu jako interní poznámku

Integrace probíhá pomocí Zammad Webhooku → Flask endpoint /zammad.

🧩 Jak integrace funguje

Zákazník nebo uživatel vytvoří ticket v Zammadu

Zammad pošle webhook (JSON payload) na:

http://<LM_HELPER_HOST>:5001/zammad


LM Helper:

vezme title + první zprávu ticketu

provede RAG vyhledávání ve FAISS

zavolá lokální LLM

vygeneruje technický návrh řešení

Výsledek se zapíše zpět do ticketu jako Internal Note

⚙️ Nastavení Zammad (krok za krokem)
1️⃣ Vytvoření API tokenu

V Zammadu:

Settings → Security → Personal Access Tokens


Name: lmhelper

Permissions (minimální):

ticket.agent

ticket.article

ticket.read

Expiration: dle potřeby (např. 1 rok)

➡️ Token si bezpečně ulož (zobrazí se jen jednou).

2️⃣ Vytvoření Webhooku
Settings → System → Webhooks → New


Základní nastavení:

Name: LM Helper

Endpoint URL:

http://127.0.0.1:5001/zammad


(pro lokální běh; v produkci nahraď IP / hostname)

Request method: POST

Payload format: JSON

SSL verification: dle prostředí (lokálně lze vypnout)

3️⃣ Trigger (kdy se webhook spustí)

Doporučené nastavení triggeru:

Object: Ticket

Event: Create nebo Update

Podmínka:

State is new nebo open

Action:

Execute Webhook → LM Helper

⚠️ Doporučení:
Pro první testy používej state = new, ať se webhook nespouští opakovaně.

🔐 Konfigurace .env

LM Helper NEUKLÁDÁ citlivé údaje do kódu.
Používá se soubor .env (není součástí Git repozitáře).

📄 .env (příklad)
ZAMMAD_URL=http://127.0.0.1:8080
ZAMMAD_TOKEN=PASTE_YOUR_PERSONAL_ACCESS_TOKEN_HERE


ZAMMAD_URL
URL, kde běží Zammad (Docker / VM / server)

ZAMMAD_TOKEN
Personal Access Token vytvořený v Zammadu

❗ .env přidej do .gitignore

🧪 Test webhooku (ručně)

Pro ověření funkčnosti lze webhook simulovat ručně:

curl -X POST http://127.0.0.1:5001/zammad \
  -H "Content-Type: application/json" \
  -d '{
    "ticket": {
      "id": 1,
      "title": "Disk full on production server"
    },
    "article": {
      "body": "df -h shows 100% usage"
    }
  }'


Pokud je vše správně:

RAG server vypíše zpracování v konzoli

do ticketu se zapíše Internal Note s návrhem řešení

interakce se uloží do logs/rag.log.jsonl

Configuration via .env

Tuning LLM výstupu (prompt, temperature, token budget) a prahy pro podobné incidenty se nově nastavují přes .env. Díky tomu lze chování asistenta ladit bez úprav kódu a zároveň držet interní prompting mimo veřejný repozitář.

Example .env
# Similarity filtering (FAISS distance: lower = closer match)
SIMILAR_MAX_DIST=0.85

# LLM generation parameters
LLM_TEMPERATURE=0.2
LLM_MAX_TOKENS=220
LLM_TOP_P=1.0
LLM_TIMEOUT_SECONDS=60

# System prompt template (variant A: newline escapes + {lang_instruction})
LLM_SYS_PROMPT=You are an IT operations troubleshooting assistant.\n\nYour task is to produce short, highly actionable, runbook-style remediation steps.\n\nSTRICT OUTPUT RULES:\n- Output MUST contain numbered steps only (each line starts with "1.", "2.", ...)\n- Each step must be on its own line (newline-separated)\n- Do NOT output any standalone summary lines, headers, or sections\n- Commands must be inline and prefixed with `$`\n- Do NOT use markdown, code blocks, or ``` formatting\n- Prefer real Linux / infra commands\n- Do NOT invent fictional tools or commands\n- Keep explanations minimal and technical\n- Clearly mark disruptive actions (restart, kill, delete)\n{lang_instruction}

# Zammad
ZAMMAD_URL=http://127.0.0.1:8080
ZAMMAD_TOKEN=REPLACE_WITH_REAL_TOKEN


📝 Logování

Každá interakce (search / solve / zammad) se ukládá do:

logs/rag.log.jsonl


Záznamy obsahují například:

typ operace

dotaz / preview ticketu

nalezené podobné incidenty

odpověď LLM

čas odezvy

použitý model

Logy lze později využít pro:

audit

ladění

trénink další verze modelu

budoucí „learning mode“

🤝 Autoři

Antonín Ečer — IT systémový inženýr se 30+ lety praxe

AI — pomoc s architekturou a implementací

Licence

Tento projekt je určen výhradně pro soukromé použití po předchozí domluvě s autorem.
Jakékoliv další šíření, komerční využití nebo úpravy k dalšímu publikování jsou možné pouze se souhlasem autora.

https://youtube.com/shorts/n3IcIipvlMw - lmhelper v 1
https://youtu.be/tgQnXgdJB6Y - LMHelper napojeni na Zammad opensource helpdesk
