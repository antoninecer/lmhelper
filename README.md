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

500 IT problémů

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


Pokud requirements.txt ještě nemáš, vytvořím ti ho.

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

🤝 Autoři

Antonín Ečer — IT systémový inženýr se 30+ lety praxe

AI — pomoc s architekturou a implementací

Licence

Tento projekt je určen výhradně pro soukromé použití po předchozí domluvě s autorem.
Jakékoliv další šíření, komerční využití nebo úpravy k dalšímu publikování jsou možné pouze se souhlasem autora.

