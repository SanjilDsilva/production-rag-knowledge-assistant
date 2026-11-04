import json
from dataclasses import dataclass
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

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
) -> IndexResult:
    validate_upload_content(path)
    filename = path.name
    file_hash = hash_file_bytes(path)

    existing_document_id = find_document_by_file_hash(
        client, tenant_id, user_id, file_hash
    )

    if existing_document_id is not None:
        return IndexResult(existing_document_id, "already uploaded")

    old_document_id = find_document_by_filename(client, tenant_id, user_id, filename)

    blocks = load_document(path)
    if not blocks:
        raise ValueError(f"No readable content in files: {path}")

    identity = json.dumps(
        [tenant_id, user_id, file_hash],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    document_id = str(uuid5(NAMESPACE_URL, f"production-rag:document:{identity}"))

    chunks = chunk_blocks(
        blocks,
        document_id=document_id,
        tenant_id=tenant_id,
        user_id=user_id,
    )

    if not chunks:
        raise ValueError(f"No chunks created from file: {path}")

    index_chunks(client, chunks, filename=filename, file_hash=file_hash)

    if old_document_id is not None and old_document_id != document_id:
        delete_document_chunks(client, tenant_id, user_id, old_document_id)
    return IndexResult(document_id, "indexed")
