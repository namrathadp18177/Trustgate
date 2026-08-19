"""
TrustGate Pipeline Orchestrator
----------------------------------
Pulls documents from Postgres that haven't been evaluated yet, runs them
through every validation stage, computes the combined Trust Score, and
writes the result to `validation_results`.

Run:
    python pipeline/run_pipeline.py
"""
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.connection import get_connection
from validation.structural import check_structural_validity
from validation.duplication import check_duplication
from validation.pii import check_pii
from validation.toxicity import check_toxicity
from validation.freshness import check_freshness
from validation.hallucination import check_hallucination_risk
from validation.trust_score import compute_trust_score


def fetch_unevaluated_documents(cur):
    cur.execute(
        """
        SELECT d.doc_id, d.raw_content, d.metadata, d.created_at, d.embedding
        FROM documents d
        LEFT JOIN validation_results v ON v.doc_id = d.doc_id
        WHERE v.result_id IS NULL
        """
    )
    return cur.fetchall()


def evaluate_document(cur, doc) -> dict:
    doc_id = doc["doc_id"]
    content = doc["raw_content"]
    metadata = doc["metadata"] or {}
    created_at = doc["created_at"]
    embedding = doc["embedding"]

    signals = {
        "structural": check_structural_validity(content, metadata),
        "duplication": check_duplication(cur, doc_id, embedding),
        "pii": check_pii(content),
        "toxicity": check_toxicity(content),
        "freshness": check_freshness(created_at),
        "hallucination": check_hallucination_risk(content),
    }

    result = compute_trust_score(signals)
    result["doc_id"] = doc_id
    result["signals"] = signals
    return result


def persist_result(cur, result: dict):
    signals = result["signals"]
    cur.execute(
        """
        INSERT INTO validation_results (
            doc_id, structural_score, duplication_score, pii_score,
            toxicity_score, freshness_score, hallucination_score,
            trust_score, decision, explanation, signal_detail
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            result["doc_id"],
            signals["structural"]["score"],
            signals["duplication"]["score"],
            signals["pii"]["score"],
            signals["toxicity"]["score"],
            signals["freshness"]["score"],
            signals["hallucination"]["score"],
            result["trust_score"],
            result["decision"],
            result["explanation"],
            json.dumps(signals, default=str),
        ),
    )


def run():
    with get_connection() as conn:
        cur = conn.cursor()
        docs = fetch_unevaluated_documents(cur)
        print(f"[pipeline] evaluating {len(docs)} document(s)")

        for doc in docs:
            result = evaluate_document(cur, doc)
            persist_result(cur, result)
            print(f"[pipeline] doc_id={result['doc_id']} trust_score={result['trust_score']} "
                  f"decision={result['decision']}")

        cur.close()

    print("[pipeline] done")


if __name__ == "__main__":
    run()
