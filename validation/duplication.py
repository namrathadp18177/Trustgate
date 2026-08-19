"""
Duplication signal.
Uses pgvector cosine similarity against already-ingested documents to find
the closest existing neighbor. A very close match (near-duplicate) lowers
the score since the document adds little new information and may indicate
a data pipeline error (double-ingestion, scraping the same page twice, etc).
"""

DUPLICATE_SIMILARITY_THRESHOLD = 0.95   # near-identical
NEAR_DUP_SIMILARITY_THRESHOLD = 0.85    # substantially similar


def check_duplication(cur, doc_id, embedding) -> dict:
    if embedding is None:
        return {"score": 100.0, "issues": ["no_embedding_skip_check"], "max_similarity": None}

    # pgvector cosine distance operator <=> returns distance (1 - cosine_similarity)
    cur.execute(
        """
        SELECT doc_id, 1 - (embedding <=> %s::vector) AS similarity
        FROM documents
        WHERE doc_id != %s AND embedding IS NOT NULL
        ORDER BY embedding <=> %s::vector
        LIMIT 1
        """,
        (embedding, doc_id, embedding),
    )
    row = cur.fetchone()

    if not row:
        return {"score": 100.0, "issues": [], "max_similarity": None}

    similarity = float(row["similarity"] if isinstance(row, dict) else row[1])
    issues = []
    score = 100.0

    if similarity >= DUPLICATE_SIMILARITY_THRESHOLD:
        score = 10.0
        issues.append(f"near_exact_duplicate (similarity={similarity:.3f})")
    elif similarity >= NEAR_DUP_SIMILARITY_THRESHOLD:
        score = 55.0
        issues.append(f"substantially_similar_to_existing_doc (similarity={similarity:.3f})")

    return {"score": round(score, 1), "issues": issues, "max_similarity": round(similarity, 4)}
