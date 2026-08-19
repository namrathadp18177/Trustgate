"""
Hallucination-risk signal.
Uses the Anthropic API to produce a *structured* risk assessment: given the
document content, the model is asked to identify unverifiable or internally
inconsistent factual claims and return strict JSON. If no API key is
configured (ANTHROPIC_API_KEY unset) or the call fails, falls back to a
cheap heuristic based on hedging/absolute language density, so the pipeline
remains runnable without an API key.
"""
import json
import os
import re

_HEDGE_WORDS = {"probably", "might", "could be", "allegedly", "unconfirmed", "rumored"}
_ABSOLUTE_WORDS = {"always", "never", "guaranteed", "proven", "definitely", "100%"}

SYSTEM_PROMPT = (
    "You are a fact-consistency auditor. Given a document, identify claims that are "
    "unverifiable, internally inconsistent, or stated with unjustified certainty. "
    "Respond with ONLY strict JSON, no prose, no markdown fences, in this exact shape: "
    '{"risk_score": <0-100 integer, 0=no hallucination risk, 100=severe>, '
    '"flagged_claims": ["claim 1", "claim 2"], "reasoning": "one sentence"}'
)


def _llm_assess(content: str) -> dict:
    import anthropic  # lazy import

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content[:4000]}],
    )
    text = "".join(block.text for block in response.content if hasattr(block, "text"))
    text = text.strip()
    text = re.sub(r"^```json|```$", "", text).strip()
    parsed = json.loads(text)
    return {
        "risk_score": float(parsed["risk_score"]),
        "flagged_claims": parsed.get("flagged_claims", []),
        "reasoning": parsed.get("reasoning", ""),
        "engine": "anthropic_llm",
    }


def _heuristic_assess(content: str) -> dict:
    lowered = content.lower()
    hedges = sum(1 for w in _HEDGE_WORDS if w in lowered)
    absolutes = sum(1 for w in _ABSOLUTE_WORDS if w in lowered)
    # unsupported absolute claims are the bigger risk signal
    risk_score = min(100.0, absolutes * 15 + hedges * 8)
    return {
        "risk_score": risk_score,
        "flagged_claims": [],
        "reasoning": f"heuristic: {absolutes} absolute-certainty phrase(s), {hedges} hedge phrase(s)",
        "engine": "heuristic_fallback",
    }


def check_hallucination_risk(content: str) -> dict:
    if not content or not content.strip():
        return {"score": 100.0, "issues": [], "risk_score": 0.0}

    use_llm = bool(os.environ.get("ANTHROPIC_API_KEY"))
    try:
        if not use_llm:
            raise RuntimeError("no API key configured")
        result = _llm_assess(content)
    except Exception:
        result = _heuristic_assess(content)

    score = round(100.0 - result["risk_score"], 1)
    issues = []
    if result["risk_score"] > 50:
        issues.append(f"high_hallucination_risk ({result['risk_score']:.0f}/100) via {result['engine']}")
    elif result["risk_score"] > 20:
        issues.append(f"moderate_hallucination_risk ({result['risk_score']:.0f}/100) via {result['engine']}")

    return {
        "score": score,
        "issues": issues,
        "risk_score": result["risk_score"],
        "flagged_claims": result.get("flagged_claims", []),
        "reasoning": result.get("reasoning", ""),
        "engine": result["engine"],
    }
