from production_rag import indexing
from production_rag.chunking import Chunk
from dataclasses import replace
from production_rag.ingestion import LoadedBlock
from qdrant_client import QdrantClient
from production_rag import vector_store
import pytest


def test_index_chunks_embeds_text_and_saves_matching_chunk(monkeypatch):
    chunk = Chunk(
        chunk_id="a" * 64,
        document_id="doc-1",
        position=0,
        text="Five useful words in this chunk.",
        chunker_version="recursive-v1",
        tenant_id="tenant-a",
        user_id="user-a",
        location="page 1",
        content_hash="b" * 64,
    )
    fake_vectors = [[0.0] * 384]
    captured = {}

    def fake_embed_documents(texts):
        captured["texts"] = texts
        return fake_vectors

    def fake_upsert_chunks(client, chunks, vectors, filename, file_hash, visibility):
        captured["saved"] = (client, chunks, vectors, filename, file_hash, visibility)

    monkeypatch.setattr(indexing, "embed_documents", fake_embed_documents)
    monkeypatch.setattr(indexing, "upsert_chunks", fake_upsert_chunks)

    client = object()
    indexing.index_chunks(
        client,
        [chunk],
        filename="shared.txt",
        file_hash="f" * 64,
        visibility="tenant_shared",
    )

    assert captured["texts"] == [chunk.text]
    assert captured["saved"] == (
        client,
        [chunk],
        fake_vectors,
        "shared.txt",
        "f" * 64,
        "tenant_shared",
    )


def test_same_file_twice_skips_second_processing(tmp_path, monkeypatch):
    path = tmp_path / "shared.txt"
    path.write_bytes(b"same raw bytes")
    state = {"document_id": None, "loads": 0, "indexes": 0, "visibility": None}
    block = LoadedBlock("Same useful file text.", "whole file")
    base_chunk = Chunk(
        chunk_id="a" * 64,
        document_id="placeholder",
        position=0,
        text=block.text,
        chunker_version="recursive-v1",
        tenant_id="tenant-a",
        user_id="user-a",
        location=block.location,
        content_hash="b" * 64,
    )

    def fake_find(client, tenant_id, user_id, file_hash):
        return state["document_id"]

    def fake_load(path):
        state["loads"] += 1
        return [block]

    def fake_chunk(blocks, document_id, tenant_id, user_id):
        return [
            replace(
                base_chunk,
                document_id=document_id,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        ]

    def fake_index(client, chunks, filename, file_hash, visibility):
        state["indexes"] += 1
        state["document_id"] = chunks[0].document_id
        state["visibility"] = visibility

    monkeypatch.setattr(
        indexing, "find_document_by_file_hash", fake_find, raising=False
    )
    monkeypatch.setattr(
        indexing,
        "find_document_by_filename",
        lambda client, tenant_id, user_id, filename: None,
    )
    monkeypatch.setattr(indexing, "load_document", fake_load, raising=False)
    monkeypatch.setattr(indexing, "chunk_blocks", fake_chunk, raising=False)
    monkeypatch.setattr(indexing, "index_chunks", fake_index, raising=False)

    client = object()
    first = indexing.index_document_file(
        client,
        path,
        "tenant-a",
        "user-a",
        document_id="doc-1",
        visibility="tenant_shared",
    )
    second = indexing.index_document_file(
        client, path, "tenant-a", "user-a", document_id="doc-2"
    )

    assert first.status == "indexed"
    assert second.status == "already uploaded"
    assert second.document_id == first.document_id
    assert state["loads"] == 1
    assert state["indexes"] == 1
    assert state["visibility"] == "tenant_shared"


def test_changed_file_with_same_name_replaces_old_chunks(tmp_path, monkeypatch):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        path = tmp_path / "shared.txt"

        def fake_chunk(blocks, document_id, tenant_id, user_id):
            return [
                Chunk(
                    chunk_id=document_id + "-chunk",
                    document_id=document_id,
                    position=0,
                    text=blocks[0].text,
                    chunker_version="test-v1",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    location=blocks[0].location,
                    content_hash="c" * 64,
                )
            ]

        monkeypatch.setattr(indexing, "chunk_blocks", fake_chunk)
        monkeypatch.setattr(
            indexing,
            "embed_documents",
            lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        )

        path.write_text("Old document version.", encoding="utf-8")
        old = indexing.index_document_file(
            client, path, "tenant-a", "user-a", "old-doc"
        )

        new_path = tmp_path / "random-storage-name.txt"
        new_path.write_text("New document version.", encoding="utf-8")
        new = indexing.index_document_file(
            client,
            new_path,
            "tenant-a",
            "user-a",
            document_id="new-doc",
            original_filename="shared.txt",
            visibility="tenant_shared",
        )

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        new_point = next(
            point for point in points if point.payload["document_id"] == "new-doc"
        )

        assert new_point.payload["visibility"] == "tenant_shared"
        assert new.previous_document_id == old.document_id
        assert {point.payload["document_id"] for point in points} == {
            old.document_id,
            new.document_id,
        }
        assert {point.payload["text"] for point in points} == {
            "Old document version.",
            "New document version.",
        }
    finally:
        client.close()


def test_same_file_for_two_tenants_is_indexed_separately(tmp_path, monkeypatch):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        path = tmp_path / "shared.txt"
        path.write_text("Shared file content.", encoding="utf-8")

        def fake_chunk(blocks, document_id, tenant_id, user_id):
            return [
                Chunk(
                    chunk_id=f"{tenant_id}-{user_id}-{document_id}-0",
                    document_id=document_id,
                    position=0,
                    text=blocks[0].text,
                    chunker_version="test-v1",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    location=blocks[0].location,
                    content_hash="c" * 64,
                )
            ]

        monkeypatch.setattr(indexing, "chunk_blocks", fake_chunk)
        monkeypatch.setattr(
            indexing,
            "embed_documents",
            lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        )

        first = indexing.index_document_file(
            client, path, "tenant-a", "user-a", document_id="shared-doc"
        )
        second = indexing.index_document_file(
            client, path, "tenant-b", "user-a", document_id="shared-doc"
        )

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        assert first.status == "indexed"
        assert second.status == "indexed"
        assert first.document_id == second.document_id == "shared-doc"
        assert len(points) == 2
        assert {point.payload["tenant_id"] for point in points} == {
            "tenant-a",
            "tenant-b",
        }
    finally:
        client.close()


def test_indexing_rejects_text_file_named_pdf(tmp_path):
    path = tmp_path / "fake.pdf"
    path.write_text("This is plain text, not a PDF.", encoding="utf-8")

    with pytest.raises(ValueError, match="do not match the PDF extension"):
        indexing.index_document_file(
            client=object(),
            path=path,
            tenant_id="tenant-a",
            user_id="user-a",
            document_id="fake-doc",
        )


def test_retry_after_interrupted_indexing_replaces_partial_chunks(
    tmp_path, monkeypatch
):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        path = tmp_path / "shared.txt"
        path.write_text("Complete document text.", encoding="utf-8")
        file_hash = indexing.hash_file_bytes(path)

        partial_chunk = Chunk(
            chunk_id="old-partial",
            document_id="doc-1",
            position=0,
            text="Only part saved before the interruption.",
            chunker_version="test-v1",
            tenant_id="tenant-a",
            user_id="user-a",
            location="page 1",
            content_hash="c" * 64,
        )
        vector_store.upsert_chunks(
            client,
            [partial_chunk],
            [[1.0] + [0.0] * 383],
            filename=path.name,
            file_hash=file_hash,
        )

        def fake_chunk(blocks, document_id, tenant_id, user_id):
            return [
                Chunk(
                    chunk_id="complete-chunk",
                    document_id=document_id,
                    position=0,
                    text=blocks[0].text,
                    chunker_version="test-v1",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    location=blocks[0].location,
                    content_hash="d" * 64,
                )
            ]

        monkeypatch.setattr(indexing, "chunk_blocks", fake_chunk)
        monkeypatch.setattr(
            indexing,
            "embed_documents",
            lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        )

        result = indexing.index_document_file(
            client, path, "tenant-a", "user-a", document_id="doc-1"
        )

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        assert result.status == "indexed"
        assert len(points) == 1
        assert points[0].payload["chunk_id"] == "complete-chunk"
    finally:
        client.close()


def test_indexing_failure_removes_partial_chunks(tmp_path, monkeypatch):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        path = tmp_path / "sample.txt"
        path.write_text("A small test document.", encoding="utf-8")

        def fake_chunk(blocks, document_id, tenant_id, user_id):
            return [
                Chunk(
                    chunk_id="partial-chunk",
                    document_id=document_id,
                    position=0,
                    text=blocks[0].text,
                    chunker_version="test-v1",
                    tenant_id=tenant_id,
                    user_id=user_id,
                    location=blocks[0].location,
                    content_hash="d" * 64,
                )
            ]

        monkeypatch.setattr(indexing, "chunk_blocks", fake_chunk)
        monkeypatch.setattr(
            indexing,
            "embed_documents",
            lambda texts: [[1.0] + [0.0] * 383 for _ in texts],
        )

        real_upsert = indexing.upsert_chunks

        def write_one_then_fail(client, chunks, vectors, filename, file_hash, visibility):
            real_upsert(client, chunks[:1], vectors[:1], filename, file_hash, visibility=visibility)
            raise RuntimeError("simulated indexing failure")

        monkeypatch.setattr(indexing, "upsert_chunks", write_one_then_fail)

        with pytest.raises(RuntimeError, match="simulated indexing failure"):
            indexing.index_document_file(
                client, path, "tenant-a", "user-a", document_id="doc-1"
            )

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        assert points == []
    finally:
        client.close()
