import pytest
from qdrant_client import QdrantClient
from production_rag.semantic_cache import (
    create_semantic_cache_collection,
    search_semantic_cache,
    store_semantic_cache_entry,
    purge_user_semantic_cache,
    extract_query_identifiers,
    DEFAULT_SIMILARITY_THRESHOLD,
)


@pytest.fixture
def qdrant_client():
    client = QdrantClient(":memory:")
    create_semantic_cache_collection(client)
    return client


def test_extract_query_identifiers():
    ids = extract_query_identifiers("How do I fix CVE-2021-44228 in Log4j 2.17.1?")
    assert "CVE-2021-44228" in ids
    assert "2.17.1" in ids
    assert "Log4j" in ids


def test_semantic_cache_store_and_search_hit(qdrant_client):
    vector = [0.1] * 384
    store_semantic_cache_entry(
        client=qdrant_client,
        query_vector=vector,
        question="What is the primary fix for Log4j?",
        answer="Upgrade to Log4j 2.17.1.",
        citations=[{"chunk_id": "c1", "document_name": "guide.txt", "location": "p.1"}],
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    # Identical or near-identical vector should hit
    hit = search_semantic_cache(
        client=qdrant_client,
        query_vector=vector,
        question="How do I fix Log4j?",
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
        threshold=0.9,
    )

    assert hit is not None
    assert hit.answer == "Upgrade to Log4j 2.17.1."
    assert hit.score >= 0.99
    assert len(hit.citations) == 1


def test_semantic_cache_cross_tenant_isolation(qdrant_client):
    vector = [0.1] * 384
    store_semantic_cache_entry(
        client=qdrant_client,
        query_vector=vector,
        question="What is secret A?",
        answer="Secret answer A",
        citations=[],
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    # Tenant B should NOT hit Tenant A's cache
    hit = search_semantic_cache(
        client=qdrant_client,
        query_vector=vector,
        question="What is secret A?",
        tenant_id="tenant-b",
        user_id="user-2",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    assert hit is None


def test_semantic_cache_version_invalidation(qdrant_client):
    vector = [0.1] * 384
    store_semantic_cache_entry(
        client=qdrant_client,
        query_vector=vector,
        question="What is the policy?",
        answer="Old policy",
        citations=[],
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    # After document update, document_set_version changes to v2
    hit = search_semantic_cache(
        client=qdrant_client,
        query_vector=vector,
        question="What is the policy?",
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v2",
        prompt_version="p1",
        model_name="model-x",
    )

    assert hit is None


def test_semantic_cache_identifier_mismatch_blocks_near_miss(qdrant_client):
    vector = [0.1] * 384
    store_semantic_cache_entry(
        client=qdrant_client,
        query_vector=vector,
        question="Explain mitigation for CVE-2021-44228",
        answer="Mitigation for CVE-2021-44228",
        citations=[],
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    # Query with different CVE ID must NOT hit, even if vectors are identical
    hit = search_semantic_cache(
        client=qdrant_client,
        query_vector=vector,
        question="Explain mitigation for CVE-2022-22965",
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    assert hit is None


def test_purge_user_semantic_cache(qdrant_client):
    vector = [0.1] * 384
    store_semantic_cache_entry(
        client=qdrant_client,
        query_vector=vector,
        question="Question 1",
        answer="Answer 1",
        citations=[],
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )

    purge_user_semantic_cache(qdrant_client, "tenant-a", "user-1")

    hit = search_semantic_cache(
        client=qdrant_client,
        query_vector=vector,
        question="Question 1",
        tenant_id="tenant-a",
        user_id="user-1",
        document_set_version="v1",
        prompt_version="p1",
        model_name="model-x",
    )
    assert hit is None
