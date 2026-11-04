from types import SimpleNamespace

from production_rag.citations import (
    Citation,
    build_citations,
    verify_citations,
)


def test_citation_uses_retrieved_chunk_details():
    retrieved_chunk = SimpleNamespace(
        payload={
            "chunk_id": "chunk-123",
            "filename": "security.pdf",
            "location": "page 4",
        }
    )

    citations = build_citations([retrieved_chunk])

    assert len(citations) == 1
    assert citations[0].chunk_id == "chunk-123"
    assert citations[0].document_name == "security.pdf"
    assert citations[0].location == "page 4"


def test_verifier_accepts_a_matching_retrieved_citation():
    retrieved_chunk = SimpleNamespace(
        payload={
            "chunk_id": "chunk-123",
            "filename": "security.pdf",
            "location": "page 4",
        }
    )
    citation = Citation("chunk-123", "security.pdf", "page 4")

    assert verify_citations([citation], [retrieved_chunk]) is True


def test_verifier_rejects_an_unknown_chunk_id():
    retrieved_chunk = SimpleNamespace(
        payload={
            "chunk_id": "chunk-123",
            "filename": "security.pdf",
            "location": "page 4",
        }
    )
    made_up_citation = Citation("not-retrieved", "security.pdf", "page 4")

    assert verify_citations([made_up_citation], [retrieved_chunk]) is False


def test_verifier_rejects_changed_source_details():
    retrieved_chunk = SimpleNamespace(
        payload={
            "chunk_id": "chunk-123",
            "filename": "security.pdf",
            "location": "page 4",
        }
    )
    altered_citation = Citation("chunk-123", "fake.pdf", "page 4")

    assert verify_citations([altered_citation], [retrieved_chunk]) is False


def test_benchmark_citation_uses_document_id_without_filename():
    retrieved_chunk = SimpleNamespace(
        payload={
            "chunk_id": "chunk-cve-1",
            "document_id": "CVE-TEST-1",
            "location": "description",
            "source": "benchmark",
        }
    )

    citations = build_citations([retrieved_chunk])

    assert citations == [
        Citation(
            chunk_id="chunk-cve-1",
            document_name="CVE-TEST-1",
            location="description",
        )
    ]
