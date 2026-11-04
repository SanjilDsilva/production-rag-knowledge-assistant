from functools import lru_cache
from typing import Any

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BATCH_SIZE = 32
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


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
    prefixed_question = f"{QUERY_PREFIX}{question}"
    return _encode([prefixed_question])[0]
