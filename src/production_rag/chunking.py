from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

if TYPE_CHECKING:
    from production_rag.ingestion import LoadedBlock


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    position: int
    text: str
    chunker_version: str
    tenant_id: str
    user_id: str
    location: str
    content_hash: str


BGE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_CHUNK_SIZE = 380
DEFAULT_OVERLAP_TOKENS = 50
CHUNKER_VERSION = "recursive-v1"
MIN_CHUNK_TOKENS = 5


@lru_cache(maxsize=1)
def get_tokenizer() -> PreTrainedTokenizerBase:
    return AutoTokenizer.from_pretrained(BGE_MODEL_NAME)


def count_tokens(text: str, tokenizer: PreTrainedTokenizerBase) -> int:
    return len(tokenizer.encode(text))


def split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text)
    paragraphs = []

    for part in parts:
        paragraph = part.strip()
        if paragraph:
            paragraphs.append(paragraph)
    return paragraphs


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text)
    sentences = []

    for part in parts:
        sentence = part.strip()
        if sentence:
            sentences.append(sentence)
    return sentences


def split_oversized_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    if count_tokens(text, tokenizer) <= max_tokens:
        return [text]

    paragraphs = split_paragraphs(text)
    if len(paragraphs) > 1:
        pieces = []
        for paragraph in paragraphs:
            pieces.extend(split_oversized_text(paragraph, tokenizer, max_tokens))
        return pieces

    sentences = split_sentences(text)
    if len(sentences) > 1:
        pieces = []
        for sentence in sentences:
            pieces.extend(split_oversized_text(sentence, tokenizer, max_tokens))
        return pieces

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    usable_tokens = max_tokens - special_tokens
    if usable_tokens < 1:
        raise ValueError("max_tokens is too small")

    pieces = []
    for start in range(0, len(token_ids), usable_tokens):
        token_slice = token_ids[start : start + usable_tokens]
        piece = tokenizer.decode(token_slice).strip()
        if piece:
            pieces.append(piece)
    return pieces


def get_overlap_text(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    overlap_tokens: int,
) -> str:
    if overlap_tokens <= 0:
        return ""

    token_ids = tokenizer.encode(text, add_special_tokens=False)
    overlap_ids = token_ids[-overlap_tokens:]
    return tokenizer.decode(overlap_ids).strip()


def pack_text_pieces(
    pieces: list[tuple[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str]]:
    packed = []
    current_parts = []
    current_locations = []

    for piece, location in pieces:
        if not piece.strip():
            continue

        candidate_parts = current_parts + [piece]
        candidate_text = "\n\n".join(candidate_parts)

        if current_parts and count_tokens(candidate_text, tokenizer) > max_tokens:
            current_text = "\n\n".join(current_parts)
            location_text = "; ".join(dict.fromkeys(current_locations))
            packed.append((current_text, location_text))

            overlap = get_overlap_text(current_parts[-1], tokenizer, overlap_tokens)
            next_parts = [overlap, piece] if overlap else [piece]
            next_locations = (
                [current_locations[-1], location] if overlap else [location]
            )
            next_text = "\n\n".join(next_parts)

            if count_tokens(next_text, tokenizer) > max_tokens:
                next_parts = [piece]
                next_locations = [location]

            current_parts = next_parts
            current_locations = next_locations

        else:
            current_parts = candidate_parts
            current_locations.append(location)

    if current_parts:
        current_text = "\n\n".join(current_parts)
        location_text = "; ".join(dict.fromkeys(current_locations))
        packed.append((current_text, location_text))

    return packed


def build_chunk_texts(
    blocks: list[LoadedBlock],
    tokenizer: PreTrainedTokenizerBase,
    max_tokens: int,
    overlap_tokens: int,
) -> list[tuple[str, str]]:
    packed = []
    pending = []

    def flush_pending() -> None:
        if pending:
            packed.extend(
                pack_text_pieces(pending, tokenizer, max_tokens, overlap_tokens)
            )
            pending.clear()

    for block in blocks:
        if not block.text.strip():
            continue

        if block.block_type in {"csv_row", "table"}:
            flush_pending()
            rows = (
                block.text.splitlines() if block.block_type == "table" else [block.text]
            )
            row_pieces = []

            for row in rows:
                pieces = split_oversized_text(row, tokenizer, max_tokens)
                row_pieces.extend((piece, block.location) for piece in pieces)

            packed.extend(
                pack_text_pieces(row_pieces, tokenizer, max_tokens, overlap_tokens)
            )
        else:
            if block.block_type == "heading":
                flush_pending()

            pieces = split_oversized_text(block.text, tokenizer, max_tokens)
            pending.extend((piece, block.location) for piece in pieces)

    flush_pending()
    return packed


def chunk_blocks(
    blocks: list[LoadedBlock],
    document_id: str,
    tenant_id: str,
    user_id: str,
    tokenizer: PreTrainedTokenizerBase | None = None,
    max_tokens: int = DEFAULT_CHUNK_SIZE,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    if tokenizer is None:
        tokenizer = get_tokenizer()

    if max_tokens <= 0 or overlap_tokens < 0 or overlap_tokens >= max_tokens:
        raise ValueError("Invalid chunk size or overlap")

    if max_tokens > tokenizer.model_max_length:
        raise ValueError("max_tokens exceeds the tokenizer's model limit")

    text_locations = build_chunk_texts(blocks, tokenizer, max_tokens, overlap_tokens)
    chunks = []
    chunker_version = f"{CHUNKER_VERSION}-size-{max_tokens}-overlap-{overlap_tokens}"

    for text, location in text_locations:
        if count_tokens(text, tokenizer) < MIN_CHUNK_TOKENS:
            continue

        position = len(chunks)
        identity = json.dumps(
            [tenant_id, user_id, document_id, position],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                document_id=document_id,
                position=position,
                text=text,
                chunker_version=chunker_version,
                tenant_id=tenant_id,
                user_id=user_id,
                location=location,
                content_hash=content_hash,
            )
        )

    return chunks
