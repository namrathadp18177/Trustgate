-- TrustGate schema
-- Requires the pgvector extension (bundled in the ankane/pgvector Docker image).

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source          TEXT NOT NULL,
    raw_content     TEXT NOT NULL,
    metadata        JSONB DEFAULT '{}'::jsonb,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding       vector(384)
);

CREATE TABLE IF NOT EXISTS validation_results (
    result_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    doc_id              UUID NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    structural_score    DOUBLE PRECISION,
    duplication_score   DOUBLE PRECISION,
    pii_score           DOUBLE PRECISION,
    toxicity_score      DOUBLE PRECISION,
    freshness_score     DOUBLE PRECISION,
    hallucination_score DOUBLE PRECISION,
    trust_score         DOUBLE PRECISION NOT NULL,
    decision            TEXT NOT NULL CHECK (decision IN ('PASS', 'REVIEW', 'REJECT')),
    explanation         TEXT NOT NULL,
    signal_detail       JSONB NOT NULL DEFAULT '{}'::jsonb,
    evaluated_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_validation_doc_id ON validation_results(doc_id);
CREATE INDEX IF NOT EXISTS idx_validation_decision ON validation_results(decision);

-- IVFFlat index for approximate nearest-neighbor similarity search (duplication signal).
-- Requires ANALYZE / a populated table to be effective; fine for this project's scale.
CREATE INDEX IF NOT EXISTS idx_documents_embedding
    ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
