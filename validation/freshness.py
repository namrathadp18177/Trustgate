"""
Freshness signal.
Scores a document based on how old it is relative to a configurable decay
window. Stale data is a real trust concern for RAG systems (answering with
outdated facts), so this penalizes documents whose `created_at` is old,
missing, or unparseable.
"""
from datetime import datetime, timezone

FULL_SCORE_WINDOW_DAYS = 30     # anything newer than this: full score
DECAY_WINDOW_DAYS = 365         # score decays to floor by this age
FLOOR_SCORE = 20.0              # oldest documents still get a nonzero score


def check_freshness(created_at) -> dict:
    if created_at is None:
        return {"score": 50.0, "issues": ["missing_created_at_defaulted"], "age_days": None}

    if isinstance(created_at, str):
        try:
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        except ValueError:
            return {"score": 40.0, "issues": ["unparseable_created_at"], "age_days": None}

    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    age_days = (datetime.now(timezone.utc) - created_at).days

    if age_days < 0:
        return {"score": 60.0, "issues": ["created_at_in_future"], "age_days": age_days}

    if age_days <= FULL_SCORE_WINDOW_DAYS:
        score = 100.0
    elif age_days >= DECAY_WINDOW_DAYS:
        score = FLOOR_SCORE
    else:
        # linear decay between the full-score window and the decay window
        progress = (age_days - FULL_SCORE_WINDOW_DAYS) / (DECAY_WINDOW_DAYS - FULL_SCORE_WINDOW_DAYS)
        score = 100.0 - progress * (100.0 - FLOOR_SCORE)

    issues = [] if age_days <= FULL_SCORE_WINDOW_DAYS else [f"document_age_{age_days}_days"]
    return {"score": round(score, 1), "issues": issues, "age_days": age_days}
