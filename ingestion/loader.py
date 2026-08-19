"""
TrustGate Ingestion
---------------------
Loads raw documents (JSON lines, CSV, or plain text files) from
`sample_data/` into the `documents` table, computing an embedding for
each so the duplication-detection signal has something to compare
against.

Run:
    python ingestion/loader.py --path ../sample_data/raw_documents.jsonl
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_connection
from validation.embeddings import get_embedder


def load_jsonl(path: str):
    docs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            docs.append(json.loads(line))
    return docs


def ingest(path: str):
    docs = load_jsonl(path)
    embedder = get_embedder()

    with get_connection() as conn:
        cur = conn.cursor()
        inserted = 0
        for doc in docs:
            content = doc.get("content", "")
            embedding = embedder.encode(content) if content.strip() else None

            cur.execute(
                """
                INSERT INTO documents (source, raw_content, metadata, created_at, embedding)
                VALUES (%s, %s, %s, COALESCE(%s, now()), %s)
                RETURNING doc_id
                """,
                (
                    doc.get("source", "unknown"),
                    content,
                    json.dumps(doc.get("metadata", {})),
                    doc.get("created_at"),
                    embedding,
                ),
            )
            inserted += 1
        cur.close()

    print(f"[ingestion] inserted {inserted} documents from {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="sample_data/raw_documents.jsonl")
    args = parser.parse_args()
    ingest(args.path)
