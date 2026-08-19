# TrustGate — AI-Powered Data Trust & Validation Platform

An end-to-end data engineering pipeline that ingests raw, messy data and runs
it through a multi-stage trust and validation system before it's considered
safe for downstream use (e.g. feeding into an LLM/RAG system). Every document
is evaluated across six independent signals and combined into a single
explainable **Trust Score (0-100)** with a **PASS / REVIEW / REJECT** decision
and a human-readable explanation.

## Architecture

```
sample_data/*.jsonl
       |
       v
ingestion/loader.py  --embeds each doc (sentence-transformers)-->  Postgres + pgvector
       |
       v
pipeline/run_pipeline.py
       |
       +--> validation/structural.py      (schema/length/encoding checks)
       +--> validation/duplication.py     (pgvector cosine similarity)
       +--> validation/pii.py             (Presidio, regex fallback)
       +--> validation/toxicity.py        (HF toxic-bert, keyword fallback)
       +--> validation/freshness.py       (age-based decay)
       +--> validation/hallucination.py   (Claude structured JSON, heuristic fallback)
       |
       v
validation/trust_score.py  -->  weighted Trust Score + decision + explanation
       |
       v
validation_results table  -->  dashboard/app.py (Streamlit)
```

## Why six independent signals

Data quality isn't a single pass/fail check. A document can be structurally
perfect but leak PII, or be fresh and clean but a near-duplicate of something
already ingested. Each signal is scored independently (0-100) so failures are
attributable, then combined with weights that reflect risk severity — PII and
toxicity are weighted highest since they're safety-critical, freshness is
weighted lowest since it's a soft quality signal. A **hard floor** rule also
forces at least REVIEW if any single signal scores below 40, so one severe
issue (e.g. an SSN) can't be averaged away by otherwise-clean signals.

## Signals in detail

| Signal | Primary engine | Fallback (no extra setup needed) |
|---|---|---|
| Structural validity | custom checks (length, encoding, required metadata) | — |
| Duplication | pgvector cosine similarity vs. ingested corpus | — |
| PII | Microsoft Presidio | regex-based detector |
| Toxicity | `unitary/toxic-bert` via Hugging Face transformers | keyword heuristic |
| Freshness | age-based linear decay | — |
| Hallucination risk | Claude API, structured JSON risk assessment | absolute/hedge-language heuristic |

Every signal degrades gracefully: if Presidio's spaCy model, the toxicity
model, or `ANTHROPIC_API_KEY` aren't available, the pipeline still runs end
to end using the fallback path (logged in each result's `engine` field).

## Run with Docker Compose

```bash
export ANTHROPIC_API_KEY=sk-...   # optional — enables LLM hallucination scoring
docker compose up --build
```

This starts Postgres (with pgvector), initializes the schema, ingests the
sample documents, runs the full validation pipeline, and starts the
dashboard at http://localhost:8501.

## Run locally without Docker

```bash
pip install -r requirements.txt

# 1. Start Postgres with pgvector (or point POSTGRES_HOST at an existing one)
docker run -d -p 5432:5432 -e POSTGRES_DB=trustgate -e POSTGRES_USER=trustgate \
  -e POSTGRES_PASSWORD=trustgate ankane/pgvector

# 2. Initialize schema
python db/connection.py

# 3. Ingest sample data
python ingestion/loader.py --path sample_data/raw_documents.jsonl

# 4. Run the validation pipeline
python pipeline/run_pipeline.py

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Tests

```bash
pytest tests/ -v
```

20 unit tests cover structural validity, freshness decay, PII regex
detection, the toxicity/hallucination fallback paths, and the trust-score
combiner's weighting and hard-floor logic — all without requiring a live
Postgres instance or API keys, so they run in CI.

## Sample data

`sample_data/raw_documents.jsonl` intentionally includes a near-empty
document, a near-duplicate pair, a document with an SSN and credit card
number, a toxic message, and an over-confident claim, so a fresh run of the
pipeline demonstrates every signal firing at least once.

## Notes on resume claims

- "Validating 100K+ records": the pipeline processes documents one at a time
  today; for that scale, `run_pipeline.py` would need batching (fetch N at a
  time) and the embedding step would move to a batched `encode()` call —
  both straightforward extensions of the current structure.
- "PASS/REVIEW/REJECT workflows": implemented as actual decision logic in
  `trust_score.py`, not a placeholder — including the hard-floor override.
- Be ready to explain why duplication uses cosine similarity over embeddings
  rather than exact-match hashing (catches near-duplicates and paraphrases,
  not just byte-identical content).
