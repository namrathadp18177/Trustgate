"""
Structural validity signal.
Checks that a document has the minimum shape needed to be usable downstream:
non-empty content, reasonable length bounds, valid encoding, and required
metadata fields present. Returns a 0-100 score plus a detail dict explaining
any deductions.
"""

MIN_LENGTH = 20
MAX_LENGTH = 50_000
REQUIRED_METADATA_FIELDS = ["source_type"]


def check_structural_validity(content: str, metadata: dict) -> dict:
    score = 100.0
    issues = []

    if not content or not content.strip():
        return {"score": 0.0, "issues": ["empty_content"]}

    length = len(content)
    if length < MIN_LENGTH:
        score -= 40
        issues.append(f"content_too_short ({length} chars < {MIN_LENGTH})")
    if length > MAX_LENGTH:
        score -= 20
        issues.append(f"content_too_long ({length} chars > {MAX_LENGTH})")

    try:
        content.encode("utf-8").decode("utf-8")
    except UnicodeError:
        score -= 30
        issues.append("invalid_encoding")

    metadata = metadata or {}
    missing = [f for f in REQUIRED_METADATA_FIELDS if f not in metadata]
    if missing:
        score -= 15 * len(missing)
        issues.append(f"missing_metadata_fields: {missing}")

    # Garbage/binary-looking content heuristic: high ratio of non-printable chars
    printable_ratio = sum(1 for c in content if c.isprintable()) / max(len(content), 1)
    if printable_ratio < 0.85:
        score -= 25
        issues.append(f"low_printable_ratio ({printable_ratio:.2f})")

    score = max(0.0, min(100.0, score))
    return {"score": round(score, 1), "issues": issues}
