"""
Trust Score combiner.
Takes the six independent signal results and combines them into a single
explainable 0-100 Trust Score with a PASS/REVIEW/REJECT decision and a
human-readable explanation. Weights reflect that PII/toxicity are safety-
critical (weighted highest) while freshness is more of a soft quality signal.
"""

WEIGHTS = {
    "structural": 0.15,
    "duplication": 0.15,
    "pii": 0.25,
    "toxicity": 0.20,
    "freshness": 0.10,
    "hallucination": 0.15,
}

PASS_THRESHOLD = 80
REVIEW_THRESHOLD = 50

# Any single high-risk signal below this floor forces at least REVIEW,
# regardless of the weighted average (e.g. a doc with 1 SSN shouldn't
# PASS just because everything else scores perfectly).
HARD_FLOOR = 40


def compute_trust_score(signals: dict) -> dict:
    """
    signals: dict of {signal_name: {"score": float, "issues": [str, ...], ...}}
    """
    weighted_sum = 0.0
    total_weight = 0.0
    all_issues = []
    hard_floor_triggered = []

    for name, weight in WEIGHTS.items():
        result = signals.get(name)
        if result is None:
            continue
        score = result["score"]
        weighted_sum += score * weight
        total_weight += weight
        all_issues.extend(f"[{name}] {issue}" for issue in result.get("issues", []))
        if score < HARD_FLOOR:
            hard_floor_triggered.append(name)

    trust_score = round(weighted_sum / total_weight, 1) if total_weight else 0.0

    if hard_floor_triggered:
        decision = "REJECT" if trust_score < REVIEW_THRESHOLD else "REVIEW"
    elif trust_score >= PASS_THRESHOLD:
        decision = "PASS"
    elif trust_score >= REVIEW_THRESHOLD:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    if not all_issues:
        explanation = f"Trust score {trust_score}/100. No issues detected across all signals. Decision: {decision}."
    else:
        issue_list = "; ".join(all_issues)
        floor_note = ""
        if hard_floor_triggered:
            floor_note = f" Hard floor triggered by: {', '.join(hard_floor_triggered)}."
        explanation = f"Trust score {trust_score}/100. Issues: {issue_list}.{floor_note} Decision: {decision}."

    return {
        "trust_score": trust_score,
        "decision": decision,
        "explanation": explanation,
        "signal_scores": {name: signals[name]["score"] for name in WEIGHTS if name in signals},
    }
