import json
import requests
import chromadb
from chromadb.config import Settings

EMBED_URL = "http://127.0.0.1:9999/v1/embeddings"
JSONL_PATH = "../data/problems.jsonl"

client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="../vectordb"
))

COLLECTION = "problems"

# If exists, delete existing collection
try:
    client.delete_collection(COLLECTION)
except:
    pass

collection = client.create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"}
)

ids = []
documents = []
embeddings = []
metadatas = []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line.strip())
        text = f"{row['problem']} {row['symptoms']} {row['analysis']}"

        # Request embedding from LM Studio
        resp = requests.post(EMBED_URL, json={"input": text}).json()
        vector = resp["data"][0]["embedding"]

        ids.append(str(row["id"]))
        documents.append(text)
        embeddings.append(vector)
        metadatas.append(row)

collection.add(
    ids=ids,
    documents=documents,
    embeddings=embeddings,
    metadatas=metadatas
)

client.persist()

print("DONE — Embeddings imported into local ChromaDB!")

