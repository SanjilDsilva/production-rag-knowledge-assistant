from functools import lru_cache
from typing import Any
from collections import OrderedDict
from hashlib import sha256
from threading import Lock


MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
QUERY_CACHE_MAX_ENTRIES = 512
_query_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()
_query_cache_lock = Lock()
_query_cache_hits = 0
_query_cache_misses = 0


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        MODEL_NAME,
        revision=MODEL_REVISION,
        device="cpu",
    )


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
    global _query_cache_hits, _query_cache_misses
    prefixed_question = f"{QUERY_PREFIX}{question}"
    cache_key = sha256(prefixed_question.encode("utf-8")).hexdigest()
    with _query_cache_lock:
        cached_vector = _query_cache.get(cache_key)
        if cached_vector is not None:
            _query_cache.move_to_end(cache_key)
            _query_cache_hits += 1
            return list(cached_vector)
        _query_cache_misses += 1
    vector = tuple(_encode([prefixed_question])[0])

    with _query_cache_lock:
        _query_cache[cache_key] = vector
        _query_cache.move_to_end(cache_key)
        while len(_query_cache) > QUERY_CACHE_MAX_ENTRIES:
            _query_cache.popitem(last=False)
    return list(vector)
