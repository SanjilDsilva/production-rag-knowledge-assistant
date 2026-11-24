import re
import time
from dataclasses import dataclass
from uuid import uuid4
from qdrant_client import QdrantClient, models

SEMANTIC_CACHE_COLLECTION = "semantic_cache"
DEFAULT_SIMILARITY_THRESHOLD = 0.92
IDENTIFIER_PATTERN = re.compile(
    r"\b(?:CVE-\d{4}-\d{4,7}|\d+(?:\.\d+)+|\d+|[A-Z][a-zA-Z0-9_]+)\b"
)
STOP_IDENTIFIERS = {
    "what", "how", "why", "when", "where", "who", "which", "is", "are",
    "explain", "describe", "can", "could", "would", "should", "the", "a", "an",
}


def extract_query_identifiers(query: str) -> set[str]:
    raw = IDENTIFIER_PATTERN.findall(query)
    return {item for item in raw if item.casefold() not in STOP_IDENTIFIERS}


@dataclass(frozen=True)
class SemanticCacheHit:
    answer: str
    citations: list[dict[str, object]]
    score: float
    cached_question: str
    created_at: float


def create_semantic_cache_collection(client: QdrantClient) -> None:
    if not hasattr(client, "collection_exists"):
        return
    try:
        if not client.collection_exists(collection_name=SEMANTIC_CACHE_COLLECTION):
            client.create_collection(
                collection_name=SEMANTIC_CACHE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
            )
    except Exception:
        pass


def search_semantic_cache(
    client: QdrantClient,
    query_vector: list[float],
    question: str,
    tenant_id: str,
    user_id: str,
    document_set_version: str,
    prompt_version: str,
    model_name: str,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> SemanticCacheHit | None:
    if not hasattr(client, "collection_exists") or not hasattr(client, "query_points"):
        return None
    try:
        if not client.collection_exists(collection_name=SEMANTIC_CACHE_COLLECTION):
            return None
    except Exception:
        return None

    query_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id", match=models.MatchValue(value=tenant_id)
            ),
            models.FieldCondition(
                key="user_id", match=models.MatchValue(value=user_id)
            ),
            models.FieldCondition(
                key="document_set_version",
                match=models.MatchValue(value=document_set_version),
            ),
            models.FieldCondition(
                key="prompt_version", match=models.MatchValue(value=prompt_version)
            ),
            models.FieldCondition(
                key="model_name", match=models.MatchValue(value=model_name)
            ),
        ]
    )

    response = client.query_points(
        collection_name=SEMANTIC_CACHE_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=1,
        score_threshold=threshold,
        with_payload=True,
    )

    results = response.points

    if not results:
        return None

    top_hit = results[0]
    payload = top_hit.payload or {}
    cached_ids = set(payload.get("extracted_ids", []))
    query_ids = extract_query_identifiers(question)

    if query_ids != cached_ids:
        return None

    return SemanticCacheHit(
        answer=str(payload.get("answer", "")),
        citations=list(payload.get("citations", [])),
        score=float(top_hit.score),
        cached_question=str(payload.get("question", "")),
        created_at=float(payload.get("created_at", 0.0)),
    )


def store_semantic_cache_entry(
    client: QdrantClient,
    query_vector: list[float],
    question: str,
    answer: str,
    citations: list[dict[str, object]],
    tenant_id: str,
    user_id: str,
    document_set_version: str,
    prompt_version: str,
    model_name: str,
) -> str:
    if not hasattr(client, "upsert"):
        return ""
    try:
        create_semantic_cache_collection(client)
        point_id = str(uuid4())
        extracted_ids = list(extract_query_identifiers(question))

        client.upsert(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=query_vector,
                    payload={
                        "question": question,
                        "answer": answer,
                        "citations": citations,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "document_set_version": document_set_version,
                        "prompt_version": prompt_version,
                        "model_name": model_name,
                        "extracted_ids": extracted_ids,
                        "created_at": time.time(),
                    },
                )
            ],
            wait=False,
        )
        return point_id
    except Exception:
        return ""


def purge_user_semantic_cache(
    client: QdrantClient,
    tenant_id: str,
    user_id: str,
) -> None:
    if not hasattr(client, "delete") or not hasattr(client, "collection_exists"):
        return
    try:
        if not client.collection_exists(collection_name=SEMANTIC_CACHE_COLLECTION):
            return
        client.delete(
            collection_name=SEMANTIC_CACHE_COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id", match=models.MatchValue(value=tenant_id)
                        ),
                        models.FieldCondition(
                            key="user_id", match=models.MatchValue(value=user_id)
                        ),
                    ]
                )
            ),
        )
    except Exception:
        pass
