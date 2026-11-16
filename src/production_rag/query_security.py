import re


SUSPICIOUS_QUERY_PATTERNS = (
    r"\bignore\b.{0,40}\b(instructions?|rules?|prompt)\b",
    r"\b(reveal|show|print)\b.{0,40}\b(system prompt|hidden instructions)\b",
    r"\b(bypass|override)\b.{0,40}\b(access|security|tenant|filter)\b",
)


def is_suspicious_query(question: str) -> bool:
    normalized_question = " ".join(question.casefold().split())
    return any(
        re.search(pattern, normalized_question) for pattern in SUSPICIOUS_QUERY_PATTERNS
    )
