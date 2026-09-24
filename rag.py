"""Embedding (BGE-M3), vector store (ChromaDB), retrieval and grounded generation (Groq)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Auto-load .env so GROQ_API_KEY works without secrets.toml
_env = Path(__file__).parent / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

import chromadb

import config as C

NOT_FOUND = (
    "I couldn't find this in the SOPs I have. Try rephrasing the question, "
    "or browse the SOP library on the left."
)

SYSTEM_PROMPT = f"""You are the Store Department SOP Assistant for {C.ORG_NAME}.
Answer ONLY from the numbered SOP passages provided in the context.
Rules:
- If the passages do not contain the answer, say you could not find it in the SOPs. Never guess or use outside knowledge.
- Cite the passages you used inline as [1], [2] (the passage numbers). Every factual statement needs a citation.
- Be concise. Use short bullets for duties, steps or responsibilities, and keep the SOP's own wording for roles and figures.
- Table data must be reproduced exactly; never estimate numbers.
- Reply in the language of the question (English or Urdu)."""


# ------------------------------------------------------------------ embeddings
@lru_cache(maxsize=1)
def get_model():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(C.EMBED_MODEL)
    model.max_seq_length = C.EMBED_MAX_SEQ
    return model


def embed(texts: list[str]) -> list[list[float]]:
    """Dense, L2-normalised BGE-M3 vectors (cosine == dot product)."""
    vecs = get_model().encode(
        texts, batch_size=C.EMBED_BATCH, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


# ---------------------------------------------------------------------- chroma
@lru_cache(maxsize=1)
def get_client():
    C.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(C.CHROMA_DIR))


def get_collection():
    return get_client().get_or_create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})


def retrieve(query: str, k: int = C.TOP_K, min_score: float = C.MIN_SCORE) -> list[dict]:
    col = get_collection()
    n = col.count()
    if n == 0:
        return []
    res = col.query(
        query_embeddings=embed([query]),
        n_results=min(max(k * 2, k), n),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    for cid, doc, meta, dist in zip(
        res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        score = 1.0 - float(dist)
        if score < min_score:
            continue
        hits.append({
            "id": cid, "score": score, "body": doc.split("\n\n", 1)[-1],
            "text": doc, **meta,
        })
    return hits[:k]


# ------------------------------------------------------------------------- LLM
def _client(api_key: str):
    from groq import Groq

    return Groq(api_key=api_key)


def _history_msgs(history: list[dict], turns: int = 3) -> list[dict]:
    keep = [m for m in history if m["role"] in ("user", "assistant")][-turns * 2:]
    return [{"role": m["role"], "content": m["content"][:700]} for m in keep]


def condense(history: list[dict], question: str, api_key: str) -> str:
    """Rewrite a follow-up ('and who signs it?') into a stand-alone search query."""
    msgs = _history_msgs(history, 2)
    if not msgs:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in msgs)
    resp = _client(api_key).chat.completions.create(
        model=C.LLM_MODEL, temperature=0, max_tokens=120,
        messages=[{"role": "user", "content":
                   "Rewrite the last question as one stand-alone search query, keeping names, roles and "
                   "SOP terms. Output only the query.\n\n" + convo + f"\nuser: {question}"}],
    )
    return resp.choices[0].message.content.strip() or question


def stream_answer(question: str, hits: list[dict], history: list[dict], api_key: str,
                  temperature: float = C.LLM_TEMPERATURE):
    context = "\n\n".join(
        f"[{i}] {h['sop_title']} \u203a {h['section']}\n{h['body']}" for i, h in enumerate(hits, 1)
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *_history_msgs(history),
                {"role": "user", "content": f"SOP passages:\n{context}\n\nQuestion: {question}"}]
    stream = _client(api_key).chat.completions.create(
        model=C.LLM_MODEL, messages=messages, temperature=temperature, stream=True, max_tokens=900,
    )
    for part in stream:
        delta = part.choices[0].delta.content
        if delta:
            yield delta
