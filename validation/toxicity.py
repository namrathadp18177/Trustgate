"""
Toxicity signal.
Tries a pretrained Hugging Face toxicity classifier first; falls back to a
small keyword-based heuristic if transformers/model download isn't available
(e.g. offline environments). Score is 100 * (1 - toxicity_probability).
"""
from functools import lru_cache

_TOXIC_KEYWORDS = {
    "idiot", "stupid", "hate you", "kill yourself", "worthless", "garbage human",
}


@lru_cache(maxsize=1)
def _get_classifier():
    from transformers import pipeline  # lazy import, heavy dependency
    return pipeline("text-classification", model="unitary/toxic-bert", top_k=None)


def _keyword_fallback(content: str) -> float:
    lowered = content.lower()
    hits = sum(1 for kw in _TOXIC_KEYWORDS if kw in lowered)
    if hits == 0:
        return 0.0
    return min(0.95, 0.3 + 0.2 * hits)


def check_toxicity(content: str) -> dict:
    if not content or not content.strip():
        return {"score": 100.0, "issues": [], "toxicity_probability": 0.0}

    try:
        classifier = _get_classifier()
        results = classifier(content[:512])  # truncate for model input limits
        flat = results[0] if isinstance(results[0], list) else results
        tox_prob = max(
            (r["score"] for r in flat if r["label"].lower() in ("toxic", "toxicity")),
            default=0.0,
        )
        engine = "unitary/toxic-bert"
    except Exception:
        tox_prob = _keyword_fallback(content)
        engine = "keyword_fallback"

    score = round(100.0 * (1 - tox_prob), 1)
    issues = []
    if tox_prob > 0.5:
        issues.append(f"high_toxicity_probability ({tox_prob:.2f}) via {engine}")
    elif tox_prob > 0.2:
        issues.append(f"moderate_toxicity_probability ({tox_prob:.2f}) via {engine}")

    return {"score": score, "issues": issues, "toxicity_probability": round(tox_prob, 3), "engine": engine}
