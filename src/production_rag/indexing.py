from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient

from production_rag.chunking import Chunk, chunk_blocks
from production_rag.embeddings import embed_documents
from production_rag.ingestion import (
    hash_file_bytes,
    load_document,
    validate_upload_content,
)
from production_rag.vector_store import (
    delete_document_chunks,
    find_document_by_filename,
    find_document_by_file_hash,
    upsert_chunks,
)


@dataclass
class IndexResult:
    document_id: str
    status: str
    previous_document_id: str | None = None


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    filename: str,
    file_hash: str,
) -> None:
    if not chunks:
        return

    texts = [chunk.text for chunk in chunks]
    vectors = embed_documents(texts)
    upsert_chunks(
        client,
        chunks,
        vectors,
        filename=filename,
        file_hash=file_hash,
    )


def index_document_file(
    client: QdrantClient,
    path: Path,
    tenant_id: str,
    user_id: str,
    document_id: str,
    original_filename: str | None = None,
) -> IndexResult:
    validate_upload_content(path)
    filename = original_filename or path.name
    file_hash = hash_file_bytes(path)

    existing_document_id = find_document_by_file_hash(
        client, tenant_id, user_id, file_hash
    )

    if existing_document_id is not None and existing_document_id != document_id:
        return IndexResult(existing_document_id, "already uploaded")

    if existing_document_id == document_id:
        delete_document_chunks(client, tenant_id, user_id, document_id)

    old_document_id = find_document_by_filename(client, tenant_id, user_id, filename)

    blocks = load_document(path)
    if not blocks:
        raise ValueError(f"No readable content in files: {path}")

    chunks = chunk_blocks(
        blocks,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    if not chunks:
        raise ValueError(f"No chunks created from file: {path}")

    try:
        index_chunks(client, chunks, filename=filename, file_hash=file_hash)
    except Exception:
        delete_document_chunks(client, tenant_id, user_id, document_id)
        raise

    previous_document_id = old_document_id if old_document_id != document_id else None
    return IndexResult(document_id, "indexed", previous_document_id)
