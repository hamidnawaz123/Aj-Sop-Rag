"""Build the index: markdown -> structure-aware chunks -> BGE-M3 -> ChromaDB.

    python ingest.py --input data/sop.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import config as C
import rag
from chunker import build_chunks


def build_index(md_path: Path = C.DATA_PATH,
                progress: Callable[[int, int], None] | None = None) -> dict:
    sops, chunks = build_chunks(Path(md_path).read_text(encoding="utf-8"))
    C.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    C.SOPS_JSON.write_text(json.dumps([
        {"id": f"sop{s.idx:02d}", "title": s.title, "markdown": s.markdown,
         "n_points": s.n_points, "n_tables": s.n_tables} for s in sops
    ], ensure_ascii=False, indent=1), encoding="utf-8")

    client = rag.get_client()
    try:
        client.delete_collection(C.COLLECTION)
    except Exception:
        pass
    # Reacquire the collection after resetting it so Chroma uses a live handle.
    col = client.get_or_create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})

    total, step = len(chunks), C.EMBED_BATCH
    for i in range(0, total, step):
        batch = chunks[i:i + step]
        col.add(ids=[c["id"] for c in batch],
                embeddings=rag.embed([c["document"] for c in batch]),
                documents=[c["document"] for c in batch],
                metadatas=[c["meta"] for c in batch])
        if progress:
            progress(min(i + step, total), total)
    return {"sops": len(sops), "chunks": total,
            "tables": sum(c["meta"]["chunk_type"] == "table" for c in chunks)}


def load_sops() -> list[dict]:
    return json.loads(C.SOPS_JSON.read_text(encoding="utf-8")) if C.SOPS_JSON.exists() else []


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(C.DATA_PATH))
    args = ap.parse_args()
    print(build_index(Path(args.input), lambda d, t: print(f"  embedded {d}/{t}", end="\r")))
