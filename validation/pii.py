"""
PII detection signal.
Tries Microsoft Presidio first (as specced); falls back to a regex-based
detector if Presidio (and its spaCy model) isn't installed, so the pipeline
still runs in lighter-weight environments. Score decreases with both the
count and the sensitivity of detected PII types.
"""
import re

HIGH_RISK_TYPES = {"US_SSN", "CREDIT_CARD", "SSN", "CARD"}
MEDIUM_RISK_TYPES = {"EMAIL_ADDRESS", "EMAIL", "PHONE_NUMBER", "PHONE"}

_REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "PHONE": re.compile(r"\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
}


def _regex_detect(content: str):
    findings = []
    for pii_type, pattern in _REGEX_PATTERNS.items():
        for match in pattern.finditer(content):
            findings.append({"type": pii_type, "text_span": match.span()})
    return findings


def _presidio_detect(content: str):
    from presidio_analyzer import AnalyzerEngine  # lazy import

    analyzer = AnalyzerEngine()
    results = analyzer.analyze(text=content, language="en")
    return [{"type": r.entity_type, "score": r.score, "text_span": (r.start, r.end)} for r in results]


def check_pii(content: str) -> dict:
    if not content:
        return {"score": 100.0, "issues": [], "findings": []}

    try:
        findings = _presidio_detect(content)
        engine = "presidio"
    except Exception:
        findings = _regex_detect(content)
        engine = "regex_fallback"

    if not findings:
        return {"score": 100.0, "issues": [], "findings": [], "engine": engine}

    high_risk_count = sum(1 for f in findings if f["type"] in HIGH_RISK_TYPES)
    medium_risk_count = sum(1 for f in findings if f["type"] in MEDIUM_RISK_TYPES)
    other_count = len(findings) - high_risk_count - medium_risk_count

    score = 100.0 - (high_risk_count * 40) - (medium_risk_count * 15) - (other_count * 5)
    score = max(0.0, score)

    issues = [f"{len(findings)} PII item(s) detected via {engine}: "
              f"{high_risk_count} high-risk, {medium_risk_count} medium-risk"]

    return {
        "score": round(score, 1),
        "issues": issues,
        "findings": [f["type"] for f in findings],
        "engine": engine,
    }
