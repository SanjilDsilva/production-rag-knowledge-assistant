import re

MAX_QUESTION_CHARS = 1000

SUSPICIOUS_QUERY_PATTERNS = (
    r"\bignore\b.{0,40}\b(?:all|any|your|the|previous|prior|above)\b.{0,20}\b(?:instructions?|prompt)\b",
    r"\b(reveal|show|print)\b.{0,40}\b(system prompt|hidden instructions)\b",
    r"\b(?:you are now|act as)\b.{0,30}\b(?:admin|administrator|developer|system)\b",
)


def query_block_reason(question: str) -> str | None:
    """First-pass filter only; access filters and the fixed prompt are main defenses."""
    if len(question) > MAX_QUESTION_CHARS:
        return "question_too_long"

    normalized_question = " ".join(question.casefold().split())
    if any(
        re.search(pattern, normalized_question) for pattern in SUSPICIOUS_QUERY_PATTERNS
    ):
        return "suspicious_query_pattern"

    return None


def is_suspicious_query(question: str) -> bool:
    return query_block_reason(question) == "suspicious_query_pattern"
