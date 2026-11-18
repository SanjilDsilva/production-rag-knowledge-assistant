from production_rag.chunking import count_tokens, get_tokenizer

from difflib import SequenceMatcher
from qdrant_client import models


def estimate_prompt_tokens(messages: list[dict[str, str]]) -> int:
    prompt_parts = []

    for message in messages:
        role = message["role"]
        content = message["content"]
        prompt_parts.append(f"{role}: {content}")

    prompt_text = "\n\n".join(prompt_parts)
    tokenizer = get_tokenizer()
    return count_tokens(prompt_text, tokenizer)


NEAR_DUPLICATE_THRESHOLD = 0.9


def remove_near_duplicate_chunks(
    chunks: list[models.ScoredPoint],
) -> list[models.ScoredPoint]:
    kept_chunks = []

    for candidate in chunks:
        candidate_payload = candidate.payload or {}
        candidate_document_id = candidate_payload.get("document_id")
        candidate_text = candidate_payload.get("text")

        if not candidate_document_id or not isinstance(candidate_text, str):
            kept_chunks.append(candidate)
            continue

        normalized_candidate = " ".join(candidate_text.casefold().split())
        is_duplicate = False

        for kept in kept_chunks:
            kept_payload = kept.payload or {}
            if kept_payload.get("document_id") != candidate_document_id:
                continue

            kept_text = kept_payload.get("text")
            if not isinstance(kept_text, str):
                continue

            normalized_kept = " ".join(kept_text.casefold().split())
            similarity = SequenceMatcher(
                None,
                normalized_candidate,
                normalized_kept,
            ).ratio()

            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                is_duplicate = True
                break

        if not is_duplicate:
            kept_chunks.append(candidate)

    return kept_chunks


def select_chunks_within_context_budget(
    chunks: list[models.ScoredPoint],
    max_context_tokens: int,
) -> list[models.ScoredPoint]:
    if max_context_tokens <= 0:
        raise ValueError("Context token budget must be positive")

    tokenizer = get_tokenizer()
    selected_chunks = []
    tokens_used = 0

    for chunk in remove_near_duplicate_chunks(chunks):
        payload = chunk.payload or {}
        chunk_id = payload.get("chunk_id")
        text = payload.get("text")

        if not chunk_id or not isinstance(text, str):
            raise ValueError("Chunk is missing its ID or text")

        context_text = f"[{chunk_id}]\n{text}"
        chunk_tokens = count_tokens(context_text, tokenizer)

        if tokens_used + chunk_tokens > max_context_tokens:
            continue

        selected_chunks.append(chunk)
        tokens_used += chunk_tokens

    return selected_chunks
