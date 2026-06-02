"""ChromaDB vector store for T&C clause retrieval.

Handles collection initialisation, ingestion from tnc_chunks.jsonl,
and semantic similarity queries. Uses ChromaDB's default embedding model
(all-MiniLM-L6-v2) so no external embedding service is required.

Usage (one-time ingestion)
--------------------------
    python knowledge_base/chroma_store.py --ingest data/processed/tnc_chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import chromadb


logger = logging.getLogger(__name__)

COLLECTION_NAME = "tnc_clauses"
_CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"


def get_collection() -> chromadb.Collection:
    """Return the persistent tnc_clauses collection, creating it if needed.

    Returns
    -------
    chromadb.Collection
        The ChromaDB collection ready for queries or inserts.
    """
    client = chromadb.PersistentClient(path=str(_CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def ingest_from_jsonl(jsonl_path: str | Path) -> int:
    """Load T&C chunks from JSONL into ChromaDB.

    Skips documents already present in the collection so re-running is safe.

    Parameters
    ----------
    jsonl_path : str or Path
        Path to tnc_chunks.jsonl produced by knowledge_base/ingest.py.

    Returns
    -------
    int
        Number of documents added.
    """
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL not found: {jsonl_path}")

    collection = get_collection()
    existing_ids = set(collection.get()["ids"])

    documents, metadatas, ids = [], [], []
    for i, line in enumerate(jsonl_path.read_text(encoding="utf-8").strip().splitlines()):
        row = json.loads(line)
        doc_id = f"tnc-{i:04d}"
        if doc_id in existing_ids:
            continue
        documents.append(row["clause_text"])
        metadatas.append({
            "section": row.get("section", "Unknown Section"),
            "page_number": row.get("page_number", 0),
            "source": row.get("source", ""),
        })
        ids.append(doc_id)

    if documents:
        collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("chroma_store.ingested %d documents", len(documents))
    else:
        logger.info("chroma_store.all_documents_already_present")

    return len(documents)


def query(query_text: str, n_results: int = 3) -> list[dict]:
    """Run a semantic similarity query against the T&C collection.

    Parameters
    ----------
    query_text : str
        The complaint text or combined query string.
    n_results : int
        Number of top clauses to return.

    Returns
    -------
    list[dict]
        Each dict has ``clause_text``, ``section``, ``score`` (0–1, higher = more similar).
    """
    collection = get_collection()
    results = collection.query(query_texts=[query_text], n_results=n_results)

    clauses = []
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dist in zip(documents, metadatas, distances):
        clauses.append({
            "clause_text": doc,
            "section": meta.get("section", "Unknown Section"),
            "score": round(1 - dist, 4),  # cosine distance → similarity
        })

    logger.debug("chroma_store.query returned %d clauses", len(clauses))
    return clauses


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Ingest T&C chunks into ChromaDB.")
    parser.add_argument("--ingest", metavar="JSONL", required=True, help="Path to tnc_chunks.jsonl")
    args = parser.parse_args()

    added = ingest_from_jsonl(args.ingest)
    print(f"Ingested {added} new documents into ChromaDB collection '{COLLECTION_NAME}'.")
    print(f"Data stored at: {_CHROMA_DIR}")
