import asyncio
from importlib import util
from pydoc import doc, html, text
from urllib.parse import urldefrag, urljoin, urlparse
from bs4 import BeautifulSoup
import playwright
from playwright.async_api import async_playwright
from readability import Document
import os
import requests
import json
import re
import uuid
from openai import OpenAI
from pymilvus import connections, list_collections, Collection, CollectionSchema, FieldSchema, DataType
from langdetect import detect
from typing import List, Any
from nltk.tokenize import sent_tokenize
import math
from collections import Counter
import html2text
import numpy as np
from sentence_transformers import CrossEncoder
import tiktoken
from openai import OpenAI
import json, math
from transformers import pipeline

embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME","text-embedding-3-small")
llm_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
cross_encoder_name_or_path=os.getenv("CROSS_ENCODER_NAME_OR_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2")
"""
cross-encoder/ms-marco-MiniLM-L-6-v2
cross-encoder/nli-deberta-v3-base
cross-encoder/stsb-roberta-large
cross-encoder/nli-roberta-base
deepset/deberta-v3-base-squad2
"""
# Define collection schema
collection_name = os.getenv("COLLECTION_NAME", "knowledge_base")

# -------- CONFIG --------
MAX_PAGES = 10000       # limit to avoid infinite crawl

WAIT_BETWEEN_REQUESTS = 1   # politeness: wait seconds between requests
ALLOWED_DOMAINS = None      # restrict crawling (e.g., {"example.com"})
# ------------------------

client = OpenAI()

embedding_model = client.embeddings.create(
    model=embedding_model_name, input="test"
)
embedding_dim = len(embedding_model.data[0].embedding)
reranker = CrossEncoder(cross_encoder_name_or_path)

def meaningful_cosine_similarity(sentence:str, sent_emb, query_emb) -> float:
    # cosine similarity = dot(a, b) / (||a|| * ||b||)
    score = float(np.dot(query_emb, sent_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(sent_emb)))
    if score>0.3:
        print(f"******** meaningful  score: {score} sentence: {sentence}")
    return score

def text_entropy(text: str) -> float:
    freq = Counter(text)
    total = len(text)
    return -sum((count / total) * math.log2(count / total) for count in freq.values())

def is_low_information(text: str, threshold=3.5) -> bool:
    return text_entropy(text) < threshold

def html_to_text_markdownish(html: str, ignore_links: bool) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = ignore_links   # or False to keep links as [text](url)
    h.body_width = 0        # no line wrapping
    return h.handle(html).strip()

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))

extractor = pipeline("text2text-generation", model="Babelscape/rebel-large")

def chunk_text(text, max_chars=400):
    """
    Split a long text into overlapping chunks (~400–450 words max).
    """
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk = [], ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def extract_facts_from_text(sentences: List[str]) -> List[tuple[str, str, str]]:
    """
    Extract (subject, relation, object) triples from long text using REBEL model.
    Works on multi-paragraph input by chunking the text automatically.
    """
    all_triples = []

    for i, sentence in enumerate(sentences, 1):
        print(f"🔹 Processing chunk {i}/{len(sentences)} ({len(sentence)} chars)")
        outputs = extractor(sentence, max_length=512, truncation=True)
        generated = outputs[0]["generated_text"]

        for triplet in generated.split("<triplet>"):
            if "<subj>" in triplet and "<obj>" in triplet:
                rel = triplet.split("<subj>")[0].strip()
                subj = triplet.split("<subj>")[1].split("<obj>")[0].strip()
                obj = triplet.split("<obj>")[1].split("</triplet>")[0].strip()
                if subj and rel and obj:
                    all_triples.append((subj, rel, obj))
    return all_triples

def safe_json_loads(output: str):
    """
    Cleans up LLM output that may include markdown formatting,
    commentary, or trailing text, and safely parses JSON arrays.
    """
    if not output:
        return []

    # Remove Markdown code fences like ```json ... ```
    output = re.sub(r"^```[a-zA-Z]*\n?", "", output.strip())
    output = re.sub(r"```$", "", output.strip())

    # Sometimes the model returns text before or after the JSON — try to extract the JSON block
    match = re.search(r"\[.*\]", output, re.DOTALL)
    if match:
        output = match.group(0)

    try:
        data = json.loads(output)
        if isinstance(data, list):
            return [s.strip() for s in data if isinstance(s, str) and len(s.strip()) > 0]
        else:
            return []
    except Exception as e:
        print(f"JSON parsing failed: {e}")
        print("Raw snippet:", output[:500])
        return []

def llm_extract_factual_sentences_stream_overlap(
    text: str,
    query: str,
    context_limit_tokens=16384,
    overlap_chars=500,
):
    """
    Process a long text with a moving overlapping window.
    Each chunk overlaps with the previous one to preserve context.
    Returns a list of factual/informative sentences about the query.
    """

    chunk_size_chars = (context_limit_tokens // 2) * 4  # approx 4 chars per token

    if not text or len(text.strip()) == 0:
        return []

    chunks = []
    total_len = len(text)
    stride = chunk_size_chars - overlap_chars

    for i in range(0, total_len, stride):
        chunk = text[i : i + chunk_size_chars]

        prompt = f"""
You are a meticulous text cleaner and segmenter.

Your task is to process the following raw web page text and output only
meaningful, relevant, and self-contained chunks related to the topic "{query}".

Steps:
1. **Clean the text**:
- Remove navigation items, headers, links, ads, boilerplate, and irrelevant content.
- Keep only sentences that are factual, informative, and related to "{query}".
- Merge dependent sentences (e.g., those starting with "This", "It", "They") so each chunk makes sense on its own.

2. **Chunk the cleaned text**:
- Group the remaining sentences into coherent chunks of about **500 characters each**.
- Each chunk should represent a self-contained piece of meaningful information.
- Avoid cutting sentences in half — always break at natural sentence boundaries.

3. **Output format**:
Return the final result as a valid JSON array of clean text chunks:
["Chunk 1", "Chunk 2", "Chunk 3", ...]

WEB PAGE TEXT:
{chunk}
"""
        try:
            response = client.chat.completions.create(
                model=llm_model_name,
                messages=[
                    {"role": "system", "content": "You are a careful text cleaner and fact extractor."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=16384,
            )

            output = response.choices[0].message.content.strip()

            try:
                part = safe_json_loads(output) #json.loads(output)
                if isinstance(part, list):
                    clean = [s.strip() for s in part if len(s.strip()) > 20]
                    chunks.extend(clean)
                else:
                    print("LLM returned non-JSON output; skipping.")
            except json.JSONDecodeError:
                print("JSON parse failed; partial text kept.")
                chunks.append(output[:300])

        except Exception as e:
            print(f"Chunk processing failed: {e}")

    # Deduplicate overlapping chunks
    unique_chunks = list(dict.fromkeys(chunks))
    return unique_chunks

def llm_filter_sentences_dynamic(sentences, query, model="gpt-4o-mini", context_limit=8000):
    system_prompt = (
        f"""You are cleaning and selecting sentences about the topic "{query}".

TASK:
1. Clean each sentence:
   - Remove any HTML, Markdown, emojis, symbols, control characters, or extra whitespace.
   - Fix broken punctuation or duplicated periods.
   - Normalize text to plain, readable English (or its original language).

2. Keep only sentences that:
   - express a clear factual statement about the topic,
   - are complete sentences with at least one verb,
   - can stand alone (not fragments or titles),
   - are not questions, slogans, bullet points, or lists.

Return the cleaned and selected sentences, one per line.
Do not summarize or add commentary.
"""
    )

    max_tokens = context_limit / 2  # leave headroom for response
    filtered_sentences = []
    batch = []
    token_count = count_tokens(system_prompt, model=model)

    for s in sentences:
        s_tokens = count_tokens(s, model=model)
        if token_count + s_tokens + 10 > max_tokens:
            # Send current batch to LLM
            batch_text = "\n".join(f"<<<{x}>>>" for x in batch)
            prompt = system_prompt + "\nSentences:\n" + batch_text
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=max_tokens,
            )
            result_text = response.choices[0].message.content.strip()
            filtered_sentences += [line.strip("-• ").strip() for line in result_text.splitlines() if line.strip()]
            # Reset batch
            batch = [s]
            token_count = count_tokens(system_prompt, model=model) + s_tokens
        else:
            batch.append(s)
            token_count += s_tokens

    # Process final batch
    if batch:
        batch_text = "\n".join(f"<<<{x}>>>" for x in batch)
        prompt = system_prompt + "\nSentences:\n" + batch_text
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=800,
        )
        result_text = response.choices[0].message.content.strip()
        filtered_sentences += [line.strip("-• ").strip() for line in result_text.splitlines() if line.strip()]

    return filtered_sentences

def generate_links_via_llm(query: str):
    prompt = f"""
You are a web research assistant.

Find up to 15 recent, high-quality, publicly available web pages about the topic: "{query}".

Return only a valid JSON array of URLs — nothing else.

Format example:
[
  "https://example.com/article1",
  "https://example.org/report2"
]

Rules:
- Include only realistic, working URLs starting with http or https.
- Each must point to a real, accessible article, report, or web page.
- Do not include summaries, titles, or markdown.
- Do not invent or guess URLs.
- Do not add extra text before or after the JSON.
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1000,
    )
    links = json.loads(response.choices[0].message.content)
    return links

# =====================
# Models and Milvus
# =====================
CHUNK_MAX_LENGTH=5000

milvus_host = os.getenv("MILVUS_HOST", "localhost")
milvus_port = int(os.getenv("MILVUS_PORT", "19530"))

# 🔗 Connect to Milvus
connections.connect("default", host=milvus_host, port=milvus_port)


"""from pymilvus import utility
if utility.has_collection(collection_name):
    print(f"Dropping old collection: {collection_name}")
    utility.drop_collection(collection_name)"""

fields = [
    FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=64),
    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=embedding_dim),
    FieldSchema(name="chunk", dtype=DataType.VARCHAR, max_length=CHUNK_MAX_LENGTH),
    FieldSchema(name="start", dtype=DataType.INT64),
    FieldSchema(name="end", dtype=DataType.INT64),
    FieldSchema(name="url", dtype=DataType.VARCHAR, max_length=255),
]
# add #fragment to url (ie. https://example.com/page.html#:~:text=Hello%20world)

schema = CollectionSchema(fields, description="RAG document chunks")

conns = [Collection(name=c) for c in list_collections()]

# Create collection if not exists
if collection_name not in [c.name for c in conns]:
    collection = Collection(name=collection_name, schema=schema)
    print(f"Created collection {collection_name}")
else:
    collection = Collection(name=collection_name)

def embed_text(text: str, model="text-embedding-3-small") -> list[float]:
    """Get embedding vector from OpenAI API"""
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding


def milvus_retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Retrieve top-k most similar chunks from Milvus.
    Returns list of dicts: [{chunk, filename, score}, ...]
    """
    # Embed query
    query_vec = embed_text(query)

    # Search in Milvus
    results = collection.search(
        data=[query_vec],
        anns_field="vector",   # field name in your schema
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=k,
        output_fields=["chunk", "filename"]
    )

    # Extract hits
    hits = results[0]
    docs = [
        {
            "chunk": h.entity.get("chunk"),
            "filename": h.entity.get("filename"),
            "score": h.distance,
        }
        for h in hits
    ]
    return docs

class ChatState(dict):
    question: str
    answer: str
    sources: List[Any]

def retrieve(state: ChatState) -> ChatState:
    question = state.get("question", "")
    if not question:
        return {"answer": "No question provided.", "sources": []}

    docs = milvus_retrieve(question)
    return {"question": question, "docs": docs}

def rerank(state: ChatState) -> ChatState:
    question = state["question"]
    docs = state.get("docs", [])

    if not docs:
        return {**state, "sources": []}

    pairs = [(question, d.page_content) for d in docs]
    scores = reranker.predict(pairs).tolist()
    reranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    top_docs = [doc for doc, _ in reranked]

    return {**state, "sources": top_docs}

def generate(state: ChatState) -> ChatState:
    question = state["question"]
    if not question:
        return {"answer": "No question provided."}
    docs = state.get("sources", [])

    context = "\n\n".join([d.page_content for d in docs])
    prompt = f"""You are a helpful assistant. Use the context below to answer.
If unknown, say "I don't know".

Context:
{context}

Question: {question}
Answer:"""

    answer = client.chat.completions.create(
        model=llm_model_name,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return {**state, "answer": answer.choices[0].message.content}

def chatbot(state = ChatState()) -> ChatState:
    state = retrieve(state)
    state = rerank(state)
    state = generate(state)
    return state

class IngestState(dict):
    raw_text: str
    url: str
    chunks: List[str]
    inserted: int

def split(state: IngestState) -> IngestState:
    text = state["raw_text"]
    #s = sent_tokenize(text)
    query = "supply chain management"
    filtered_chunks = llm_extract_factual_sentences_stream_overlap(text, query, context_limit_tokens=16384)

    return {"raw_text": state["raw_text"], "chunks": filtered_chunks}

def _split(state: IngestState) -> IngestState:
    text = state["raw_text"]
    doc_language = detect(text)
    query = "supply chain management"
    s = sent_tokenize(text)
    embeddings = embedding_model.embed_documents(s)
    query_emb = np.array(embedding_model.embed_query(query))
    #sentences = [x[0] for x in zip(s, embeddings) if meaningful_cosine_similarity(x[0], x[1], query_emb) > 0.3 ]
    emb_matrix = np.array(embeddings)
    query_emb_norm = query_emb / np.linalg.norm(query_emb)
    sent_norms = emb_matrix / np.linalg.norm(emb_matrix, axis=1, keepdims=True)
    cosines = np.dot(sent_norms, query_emb_norm)
    sentences = [s[i] for i, c in enumerate(cosines) if c > 0.3]

    if not sentences:
        return {"raw_text": state["raw_text"], "chunks": []}

    pairs = [(query, s) for s in sentences]
    scores = reranker.predict(pairs).tolist()
    sentences = [x[0] for x in zip(sentences, scores) if x[1] > 0.3]
    return {"raw_text": state["raw_text"], "chunks": sentences}

def embed(state: IngestState) -> IngestState:
    """Embed chunks and insert into Milvus"""
    texts = state.get("chunks", [])
    if not texts:
        return {"raw_text": state["raw_text"], "url": state["url"], "chunks": [], "inserted": 0}

    embeddings = embedding_model.embed_documents(texts)

    entities = [
        [str(uuid.uuid4()) for _ in texts],
        embeddings,
        texts,
        [0 for _ in texts],
        [len(t) for t in texts],
        [state.get("url", "unknown") for _ in texts],  # source field
    ]
    collection.insert(entities)
    collection.flush()

    return {"raw_text": state["raw_text"], "url": state["url"], "chunks": texts, "inserted": len(texts)}


def ingester(state = IngestState()) -> IngestState:
    state = split(state)
    state = embed(state)
    return state

def should_visit(base_url, href):
    if not href:
        return None

    # Ignore javascript, mailto, tel, etc.
    if href.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None

    # Make absolute URL
    url = urljoin(base_url, href)

    # Remove fragment (#part)
    url = urldefrag(url).url

    # Parse for comparisons
    parsed_base = urlparse(base_url)
    parsed_new = urlparse(url)

    # Skip if same page (same path, no query change)
    if (parsed_base.netloc == parsed_new.netloc and
        parsed_base.path == parsed_new.path and
        parsed_base.query == parsed_new.query):
        return None

    return url

import time

MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
    ".webp", ".ico", ".tiff", ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".mp3", ".wav", ".ogg", ".flac",
    ".pdf",  # optionally skip PDFs if you want only HTML pages
}

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urldefrag

# File types we don't want to crawl
MEDIA_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".svg",
    ".webp", ".ico", ".tiff", ".mp4", ".mov", ".avi",
    ".mkv", ".webm", ".mp3", ".wav", ".ogg", ".flac",
    ".pdf", ".zip", ".tar", ".gz", ".rar"
}

def extract_valid_links(html: str, base_url: str = "") -> list[str]:
    """
    Extracts valid, absolute HTTP(S) links from HTML.
    Filters out:
      - non-http(s) URLs (mailto:, javascript:)
      - media/binary files
      - same-page anchors (#)
      - self-referencing URLs (same as base_url)
    """
    soup = BeautifulSoup(html, "html.parser")
    base_parsed = urlparse(base_url)
    valid_links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href or href.startswith(("javascript:", "mailto:")):
            continue

        # Resolve relative to absolute and remove fragments
        absolute_url = urljoin(base_url, href)
        clean_url = urldefrag(absolute_url).url  # removes #anchors

        # Skip if non-http(s)
        if not clean_url.lower().startswith(("http://", "https://")):
            continue

        # Skip media/binary extensions
        if any(clean_url.lower().endswith(ext) for ext in MEDIA_EXTENSIONS):
            continue

        # Skip self-links (same page or same base path)
        parsed = urlparse(clean_url)
        if parsed.netloc == base_parsed.netloc and parsed.path == base_parsed.path:
            continue

        valid_links.add(clean_url)

    return valid_links

async def search(playwright, query):
    visited = set()
    to_visit = []
    #browser = await playwright.chromium.launch(headless=True)
    browser = await playwright.chromium.launch(headless=False)  # or True if needed
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        java_script_enabled=True,
    )
    """context = await browser.new_context(
        locale="en-US",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"}
    )"""
    page = await context.new_page()


    async def accept_cookies(page):
        """Try to click any cookie consent button found on the page."""
        selectors = [
            "button:has-text('ACCEPT ALL COOKIES')",
            "button:has-text('Accept all')",
            "button:has-text('Accept All')",
            "button:has-text('I agree')",
            "button:has-text('I Agree')",
            "button:has-text('Agree')",
            "button:has-text('Allow all')",
            "button:has-text('Allow All Cookies')",
            "button:has-text('Accept cookies')",
            "button:has-text('Accept Cookies')",
            "button:has-text('Consent')",
            "button:has-text('OK')",
            "button:has-text('Got it')",
            "button:has-text('Tout accepter')",      # French
            "button:has-text('Authoriser tous les cookies')",      # French
            "button:has-text('Accepter tout')",       # French variant
            "button:has-text('Alles akzeptieren')",   # German
            "button:has-text('Aceptar todo')",        # Spanish
            "button:has-text('Aceptar todas')",       # Spanish variant
            "button:has-text('Accetta tutti')",       # Italian
            "button:has-text('Agree & continue')",
            "button:has-text('Continue without accepting')",  # fallback
        ]

        for selector in selectors:
            try:
                button = page.locator(selector)
                if await button.count() > 0:
                    await button.first.click(timeout=2000)
                    return True
            except Exception:
                continue
        return False
    
    async def fetch_page(url):
        """Fetch a page, render JavaScript, extract text + links."""
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=10000)
        except:
            return
        
        await accept_cookies(page)

        print(f"Fetched: {url}")
        # Extract visible text
        html = await page.content()   # rendered HTML

        with open("page.html", "w", encoding="utf-8") as f:
            f.write(html)

        links = extract_valid_links(html)
        text = html_to_text_markdownish(html, ignore_links=True)

        for link in links:
            parsed = urlparse(link)
            if ALLOWED_DOMAINS and parsed.netloc not in ALLOWED_DOMAINS:
                continue
            if link not in to_visit and link not in visited:
                to_visit.append(link)

        start = time.time()
        result = ingester(state={"raw_text": text, "url": url})
        print(f"inserted {result.get('inserted', 0)} chunks, to visit: {len(to_visit)}, visited: {len(visited)} elapsed ingestion: {time.time() - start:.4f}s")

    async def crawl(url):
        """Recursive crawler function."""
        if url in visited:
            return

        visited.add(url)
        await fetch_page(url)

        # politeness delay
        #await asyncio.sleep(WAIT_BETWEEN_REQUESTS)

    for link in generate_links_via_llm(query):
        to_visit.append(link)

    try:
        while to_visit and len(visited) < MAX_PAGES:
            await crawl(to_visit.pop(0))
    finally:
        await browser.close()

async def start():
    print("Starting search...")
    query = "supply chain management"
    async with async_playwright() as p:
        await search(p, query)


if __name__ == "__main__":
    asyncio.run(start())
