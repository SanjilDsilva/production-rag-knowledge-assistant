from functools import lru_cache
from typing import Any
from hashlib import sha256
import json
import os
from pathlib import Path
from production_rag.answer_cache import (
    BoundedAnswerCache,
    CacheBackend,
    create_cache_backend,
)
import logging
from time import perf_counter

logger = logging.getLogger(__name__)

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_CACHE_MAX_ENTRIES = 512


def create_query_cache() -> CacheBackend[list[float]]:
    return create_cache_backend(
        os.getenv("RAG_CACHE_BACKEND", "memory"),
        namespace="query_embeddings",
        max_entries=QUERY_CACHE_MAX_ENTRIES,
        ttl_seconds=900,
        database_path=Path(os.getenv("RAG_DATA_DIR", "data")) / "cache.db",
    )


_query_cache: CacheBackend[list[float]] = create_query_cache()


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )

def is_embedding_model_loaded() -> bool:
    return get_embedding_model.cache_info().currsize > 0

def _encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []

    encoded_vectors = get_embedding_model().encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return [[float(value) for value in vector] for vector in encoded_vectors]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _encode(texts)


def embed_query(question: str) -> list[float]:
    normalized_question = " ".join(question.casefold().split())
    prefixed_question = f"{QUERY_PREFIX}{normalized_question}"
    key_parts = {
        "question": normalized_question,
        "model_revision": MODEL_REVISION,
        "query_instruction": QUERY_PREFIX,
    }
    cache_key_material = json.dumps(
        key_parts, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    cache_key = sha256(cache_key_material).hexdigest()
    lookup_started_at = perf_counter()
    cached_vector = _query_cache.get(cache_key)
    cache_lookup_ms = (perf_counter() - lookup_started_at) * 1000
    cache_stats = _query_cache.stats()

    logger.info(
        "query_embedding_cache_lookup",
        extra={
            "cache_backend": _query_cache.backend_name,
            "cache_hit": cached_vector is not None,
            "cache_lookup_ms": round(cache_lookup_ms, 4),
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "cache_hit_rate": cache_stats["hit_rate"],
        },
    )
    if cached_vector is not None:
        return list(cached_vector)

    vector = _encode([prefixed_question])[0]
    _query_cache.set(cache_key, vector)
    return list(vector)
