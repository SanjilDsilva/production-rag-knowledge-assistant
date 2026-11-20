from dataclasses import dataclass, field

from qdrant_client import models


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    document_name: str
    location: str
    text: str | None = field(default=None, compare=False)


def build_citations(
    retrieved_chunks: list[models.ScoredPoint],
) -> list[Citation]:
    citations = []

    for retrieved_chunk in retrieved_chunks:
        payload = retrieved_chunk.payload or {}

        chunk_id = payload.get("chunk_id")
        document_name = payload.get("filename")
        if document_name is None and payload.get("source") == "benchmark":
            document_name = payload.get("document_id")
        location = payload.get("location")
        text = payload.get("text")

        if not chunk_id or not document_name or not location:
            raise ValueError("Retrieved chunk is missing citation details")

        citations.append(
            Citation(
                chunk_id=chunk_id,
                document_name=document_name,
                location=location,
                text=text,
            )
        )

    return citations


def verify_citations(
    citations: list[Citation],
    retrieved_chunks: list[models.ScoredPoint],
) -> bool:
    valid_citations = {
        citation.chunk_id: citation for citation in build_citations(retrieved_chunks)
    }

    return all(
        valid_citations.get(citation.chunk_id) == citation for citation in citations
    )
