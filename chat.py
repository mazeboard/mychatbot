import os
import uuid
import zipfile
import tempfile
import fitz  # PyMuPDF
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from azure.storage.blob import BlobServiceClient
from nltk.tokenize import sent_tokenize
from pymilvus import (
    connections,
    list_collections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
)

from openai import OpenAI
from sentence_transformers import CrossEncoder
from llama_cpp import Llama
from langdetect import detect

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME","text-embedding-3-small")
llm_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
cross_encoder_name_or_path=os.getenv("CROSS_ENCODER_NAME_OR_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# =====================
# FastAPI setup
# =====================
chat = FastAPI(title="RAG Chatbot")

# =====================
# Milvus connection
# =====================
milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
connections.connect("default", host=milvus_host, port=milvus_port)

collection_name = "knowledge_base"
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Create embedding vector dimension dynamically
test_embedding = openai_client.embeddings.create(
    model=embedding_model_name, input="test"
)
embedding_dim = len(test_embedding.data[0].embedding)

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=2000),
    FieldSchema(name="start", dtype=DataType.INT64),
    FieldSchema(name="end", dtype=DataType.INT64),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=255),
]
schema = CollectionSchema(fields, description="RAG document chunks")

collection = Collection(name=collection_name)

# =====================
# Models
# =====================
reranker = CrossEncoder(cross_encoder_name_or_path)

# =====================
# Embedding + Retrieval Utils
# =====================
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings directly via OpenAI API"""
    res = openai_client.embeddings.create(model=embedding_model_name, input=texts)
    return [d.embedding for d in res.data]


def milvus_search(query: str, k=5):
    """Perform a similarity search in Milvus"""
    query_vec = embed_texts([query])[0]
    results = collection.search(
        data=[query_vec],
        anns_field="vector",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["chunk", "url"],
    )
    hits = results[0]
    docs = [{"page_content": h.entity.get("chunk"), "url": h.entity.get("url")} for h in hits]
    return docs


# =====================
# Chatbot Pipeline
# =====================
from tiktoken import encoding_for_model

def build_context_within_limit(filtered_docs, prompt: str, max_window: int=128000):
    """
    Concatenates documents until the total token count stays below the LLM context window.
    """
    enc = encoding_for_model(llm_model_name)
    
    # Estimate token count for the static parts (prompt + question)
    base_tokens = len(enc.encode(prompt))
    
    # Keep adding docs until limit
    context_docs = []
    total_tokens = base_tokens
    for (d, s) in filtered_docs:
        doc_tokens = len(enc.encode(d["page_content"]))
        if total_tokens + doc_tokens > max_window - 500:  # leave buffer for the answer
            break
        context_docs.append((d, s))
        total_tokens += doc_tokens

    return context_docs

def run_chatbot(question: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    if not question:
        return {"answer": "No question provided.", "sources": []}

    prompt = """You are a helpful assistant. Use the context below to answer.
If unknown, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    # Retrieve from Milvus
    docs = milvus_search(question, k=10)
    if not docs:
        return {"answer": "No relevant documents found.", "sources": []}

    # Rerank
    pairs = [(question, d["page_content"]) for d in docs]
    scores = reranker.predict(pairs).tolist()
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    # Keep only those above threshold
    threshold = 0.8
    filtered = [(d, s) for d, s in reranked if s > threshold]
    top_docs = build_context_within_limit(filtered, prompt)

    # Generate
    context = "\n\n".join([d["page_content"] for (d, _) in top_docs])

    completion = openai_client.chat.completions.create(
        model=llm_model_name,
        messages=[{"role": "user", "content": prompt.format(context=context, question=question)}],
        temperature=0.2,
    )
    answer = completion.choices[0].message.content
    result = {
        "answer": answer,
        "sources": [
            {
                "content": d.get("page_content"),
                "url": d.get("url"),
                "score": float(s)
            }
            for (d, s) in top_docs
        ],
        "session_id": session_id or str(uuid.uuid4())
    }
    return result


# =====================
# FastAPI Endpoints
# =====================
class QueryRequest(BaseModel):
    session_id: Optional[str] = None
    question: str


@chat.post("/search")
async def search(req: QueryRequest):
    return run_chatbot(req.question, req.session_id)


@chat.get("/health")
async def health():
    return {"status": "ok", "collection": "documents"}

# Path to React build
build_dir = os.path.join(os.path.dirname(__file__), "build")

# Mount static assets (JS, CSS, etc.)
chat.mount("/static", StaticFiles(directory=os.path.join(build_dir, "static")), name="static")

# Serve React index.html at root
@chat.get("/{full_path:path}")
async def serve_react(full_path: str):
    return FileResponse(os.path.join(build_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run("chat:chat", host="0.0.0.0", port=8000, reload=False)
