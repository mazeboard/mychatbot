from fastapi import FastAPI, UploadFile
from pymilvus import connections, list_collections, Collection, CollectionSchema, FieldSchema, DataType
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import fitz  # PyMuPDF for PDF
import numpy as np
import uuid
from pydantic import BaseModel
from sentence_transformers import CrossEncoder
import uvicorn
from openai import OpenAI
from pymilvus import Index
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag
from readability import Document


client = OpenAI()

# 🔗 Connect to Milvus
connections.connect("default", host="localhost", port="19530")

# Define collection schema
collection_name = "documents"
dim = 384  # depends on embedding model (MiniLM has 384)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="start", dtype=DataType.INT64),
    FieldSchema(name="end", dtype=DataType.INT64),
]

schema = CollectionSchema(fields, description="RAG document chunks")

conns = [Collection(name=c) for c in list_collections()]



# Create collection if not exists
if collection_name not in [c.name for c in conns]:
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection {collection_name}")
else:
    collection = Collection(name=collection_name)

# After inserting data
index_params = {
    "index_type": "IVF_FLAT",  # or HNSW, ANNOY, etc.
    "metric_type": "IP",
    "params": {"nlist": 128},
}

idx = Index(collection, "embedding", index_params=index_params)

async def lifespan(app: FastAPI):
    # Startup code
    collection.load()
    print(f"Collection {collection_name} loaded into memory")
    yield  # here the app runs
    # Shutdown code
    collection.release()
    print(f"Collection {collection_name} released from memory")

app = FastAPI(lifespan=lifespan)

# Embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# --- Utilities ---
def _split_text(text, max_tokens=100):
    sentences = sent_tokenize(text)
    chunks = []
    chunk = ""
    chunk_tokens = 0
    start_idx = 0
    current_idx = 0

    for sent in sentences:
        tokens = len(sent.split())
        if chunk_tokens + tokens > max_tokens:
            end_idx = current_idx
            if chunk.strip():
                chunks.append((start_idx, end_idx, chunk.strip()))
            chunk = sent
            chunk_tokens = tokens
            start_idx = current_idx
        else:
            if chunk:
                chunk += " " + sent
            else:
                chunk = sent
            chunk_tokens += tokens
        current_idx += len(sent) + 1
    if chunk:
        chunks.append((start_idx, len(text), chunk.strip()))
    return chunks

def split_text(text):
    sentences = sent_tokenize(text)
    chunks = []
    idx = 0

    for sent in sentences:
        end_idx = idx + len(sent)
        chunks.append((idx, end_idx, sent))
        idx = end_idx
    return chunks

def embed(texts):
    return embed_model.encode(texts, convert_to_numpy=True)

visited=set()

def crawl(url, max_depth=2, depth=0):
    if depth > max_depth or url in visited:
        return []
    visited.add(url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []

    # Extract readable text
    doc = Document(response.text)
    text = doc.summary()
    soup = BeautifulSoup(text, "html.parser")
    page_text = soup.get_text(separator=" ", strip=True)

    results = [(url, page_text)]

    # Extract links
    soup_links = BeautifulSoup(response.text, "html.parser")
    for link in soup_links.find_all("a", href=True):
        next_url = urljoin(url, link["href"])
        next_url = urldefrag(next_url).url  # remove #anchors
        if next_url.startswith("http"):
            results.extend(crawl(next_url, max_depth, depth + 1))

    return results

API_KEY = "AIzaSyALuDpzOPdjCt82Ey9qeMurruHYhR_kOqU"
CSE_ID = "64e176914e0b7409a"

def google_search(query, num=5):
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": API_KEY,
        "cx": CSE_ID,
        "q": query,
        "num": num
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    results = []
    for item in data.get("items", []):
        results.append({
            "title": item.get("title"),
            "link": item.get("link"),
            "snippet": item.get("snippet")
        })
    return results

# --- API ---
class GoogleQueryRequest(BaseModel):
    query: str
    num: int = 5

@app.post("/search_google")
async def search_google(req: GoogleQueryRequest):
    query = req.query
    num = req.num
    results = google_search(query, num)
    return {"query": query, "results": results}

@app.post("/ingest")
async def ingest(file: UploadFile):
    if file.filename.endswith(".pdf"):
        pdf = fitz.open(stream=await file.read(), filetype="pdf")
        text = "\n".join(page.get_text() for page in pdf)
    else:
        text = (await file.read()).decode("utf-8")

    chunks = split_text(text)
    texts = [c[2] for c in chunks]
    embeddings = embed(texts)

    entities = [
        [str(uuid.uuid4()) for _ in chunks],             # id
        embeddings.tolist(),                             # embedding
        texts,                                           # text
        [c[0] for c in chunks],                          # start
        [c[1] for c in chunks],                          # end
    ]

    collection.insert(entities)
    collection.flush()
    """Milvus 2.x allows you to keep a collection loaded and insert new entities. However, searches will only see flushed data:"""
    # collection.load()       # reload collection into memory with the new entities
    return {"status": "ok", "chunks": len(chunks)}

# Define request model
class QueryRequest(BaseModel):
    question: str
    top_k: int = 5

# Load reranker once
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def build_prompt(question: str, contexts: list[str]) -> str:
    """
    Construct a grounded prompt for the LLM using retrieved chunks.
    """
    context_text = "\n\n".join([f"Context {i+1}: {c}" for i, c in enumerate(contexts)])
    prompt = f"""
You are an assistant for question answering. Use the following context to answer the question.
If the answer cannot be found in the context, say "I don't know" instead of guessing.

Context:
{context_text}

Question: {question}

Answer:
"""
    return prompt.strip()

def generate_answer(question: str, contexts: list[str], model="gpt-4o-mini") -> str:
    prompt = build_prompt(question, contexts)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

@app.post("/search")
async def search(req: QueryRequest):
    question = req.question
    top_k = req.top_k or 5

    # Step 1: embed and retrieve from Milvus
    query_vec = embed_model.encode([question], convert_to_numpy=True)[0]
    results = collection.search(
        data=[query_vec.tolist()],
        anns_field="embedding",
        param={"metric_type": "IP", "params": {"nprobe": 10}},
        limit=top_k * 3,
        output_fields=["chunk"]
    )
    chunks = [hit.entity.get("chunk") for hit in results[0]]

    # Step 2: rerank (optional)
    pairs = [(question, chunk) for chunk in chunks]
    scores = reranker.predict(pairs).tolist()
    reranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
    top_chunks = [chunk for chunk, score in reranked[:top_k]]

    # Step 3: build prompt + query LLM
    answer = generate_answer(question, top_chunks)

    sources = [{
        "doc": hit.entity.get("id"),  # or another field if you store document name
        "snippet": hit.entity.get("chunk")[:300],  # snippet of the text
        "score": float(hit.score)  # Milvus similarity score
        }
        for hit in results[0]
    ]

    # Use reranker scores for confidence
    confidence = max(scores) if scores else 0.0

    return {
        "answer": answer,
        "sources": sources,
        "confidence": float(confidence)
    }

@app.get("/health")
async def health():
    return {"status": "ok", "indexed": (embed_array.shape[0] if embed_array is not None else 0)}

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Adjust as needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
