"""
Unit tests for TrustGate's validation signals. These test the pure-function
signals (structural, freshness, PII regex fallback, trust score combiner)
without requiring a live Postgres instance, so they run in CI easily.
Duplication (needs pgvector) and toxicity/hallucination (need model/API
access) are covered by fallback-path tests only.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from validation.structural import check_structural_validity
from validation.freshness import check_freshness
from validation.pii import _regex_detect, check_pii
from validation.toxicity import _keyword_fallback
from validation.hallucination import _heuristic_assess
from validation.trust_score import compute_trust_score


class TestStructuralValidity:
    def test_empty_content_scores_zero(self):
        result = check_structural_validity("", {"source_type": "test"})
        assert result["score"] == 0.0
        assert "empty_content" in result["issues"]

    def test_valid_content_scores_high(self):
        content = "This is a perfectly reasonable piece of content for testing purposes here."
        result = check_structural_validity(content, {"source_type": "test"})
        assert result["score"] == 100.0

    def test_missing_metadata_penalized(self):
        content = "This is a perfectly reasonable piece of content for testing purposes here."
        result = check_structural_validity(content, {})
        assert result["score"] < 100.0
        assert any("missing_metadata_fields" in i for i in result["issues"])

    def test_too_short_penalized(self):
        result = check_structural_validity("short", {"source_type": "test"})
        assert result["score"] < 100.0


class TestFreshness:
    def test_recent_document_full_score(self):
        recent = datetime.now(timezone.utc) - timedelta(days=1)
        result = check_freshness(recent)
        assert result["score"] == 100.0

    def test_old_document_decays(self):
        old = datetime.now(timezone.utc) - timedelta(days=400)
        result = check_freshness(old)
        assert result["score"] == 20.0

    def test_missing_created_at_defaults(self):
        result = check_freshness(None)
        assert result["score"] == 50.0

    def test_iso_string_parsed(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        result = check_freshness(recent)
        assert result["score"] == 100.0


class TestPII:
    def test_regex_detects_email(self):
        findings = _regex_detect("contact me at test@example.com please")
        assert any(f["type"] == "EMAIL" for f in findings)

    def test_regex_detects_ssn(self):
        findings = _regex_detect("SSN: 123-45-6789")
        assert any(f["type"] == "SSN" for f in findings)

    def test_clean_content_scores_100(self):
        result = check_pii("This document has no sensitive information at all.")
        assert result["score"] == 100.0

    def test_ssn_heavily_penalized(self):
        result = check_pii("SSN on file: 123-45-6789")
        assert result["score"] < 70.0


class TestToxicityFallback:
    def test_clean_text_zero_probability(self):
        assert _keyword_fallback("This is a lovely and helpful message.") == 0.0

    def test_toxic_keywords_detected(self):
        prob = _keyword_fallback("You are such an idiot and this is worthless garbage.")
        assert prob > 0.0


class TestHallucinationHeuristic:
    def test_absolute_language_raises_risk(self):
        result = _heuristic_assess("This always works and is guaranteed and proven 100% of the time.")
        assert result["risk_score"] > 0

    def test_neutral_text_low_risk(self):
        result = _heuristic_assess("The report summarizes quarterly sales figures by region.")
        assert result["risk_score"] == 0


class TestTrustScoreCombiner:
    def _perfect_signals(self):
        return {
            "structural": {"score": 100.0, "issues": []},
            "duplication": {"score": 100.0, "issues": []},
            "pii": {"score": 100.0, "issues": []},
            "toxicity": {"score": 100.0, "issues": []},
            "freshness": {"score": 100.0, "issues": []},
            "hallucination": {"score": 100.0, "issues": []},
        }

    def test_all_perfect_scores_pass(self):
        result = compute_trust_score(self._perfect_signals())
        assert result["trust_score"] == 100.0
        assert result["decision"] == "PASS"

    def test_low_pii_score_forces_review_or_reject(self):
        signals = self._perfect_signals()
        signals["pii"] = {"score": 10.0, "issues": ["near_exact_duplicate"]}
        result = compute_trust_score(signals)
        assert result["decision"] in ("REVIEW", "REJECT")

    def test_all_zero_rejects(self):
        signals = {k: {"score": 0.0, "issues": ["bad"]} for k in self._perfect_signals()}
        result = compute_trust_score(signals)
        assert result["decision"] == "REJECT"
        assert result["trust_score"] == 0.0

    def test_explanation_includes_decision(self):
        result = compute_trust_score(self._perfect_signals())
        assert "PASS" in result["explanation"]
