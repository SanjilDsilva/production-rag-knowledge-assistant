import json

import pytest

from production_rag.benchmark_data import load_benchmark_artifacts
import numpy as np
from qdrant_client import QdrantClient, models

from pathlib import Path

from production_rag import benchmark_data, vector_store
from production_rag.chunking import Chunk


def test_manifest_rejects_changed_vector_file(tmp_path):
    (tmp_path / "vectors.npy").write_bytes(b"some vector bytes")
    (tmp_path / "chunks.jsonl").write_text("{}\n", encoding="utf-8")

    manifest = {
        "files": {
            "vectors.npy": "wrong fingerprint",
            "chunks.jsonl": "some fingerprint",
        }
    }
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Checksum mismatch for vectors.npy"):
        load_benchmark_artifacts(tmp_path)


def test_benchmark_import_is_idempotent_and_search_is_filtered(
    tmp_path,
    monkeypatch,
):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    vectors = np.zeros((1, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    record = {
        "chunk_id": "a" * 64,
        "document_id": "CVE-TEST-1",
        "position": 0,
        "text": "A test vulnerability record.",
        "chunker_version": "recursive-v1-size-380-overlap-50",
        "tenant_id": "benchmark-tenant",
        "user_id": "benchmark-user",
        "location": "description",
        "content_hash": "b" * 64,
    }
    monkeypatch.setattr(
        benchmark_data,
        "load_benchmark_artifacts",
        lambda directory: (vectors, [record]),
    )

    try:
        first_count = benchmark_data.import_benchmark_artifacts(client, tmp_path)
        second_count = benchmark_data.import_benchmark_artifacts(client, tmp_path)

        assert first_count == second_count == 1

        benchmark_results = vector_store.search_chunks(
            client,
            "test vulnerability",
            vectors[0].tolist(),
            "benchmark-tenant",
            "benchmark-user",
            search_mode="dense",
        )
        other_tenant_results = vector_store.search_chunks(
            client,
            "test vulnerability",
            vectors[0].tolist(),
            "tenant-other",
            "user-other",
            search_mode="dense",
        )

        other_user_results = vector_store.search_chunks(
            client,
            "test_vulnerability",
            vectors[0].tolist(),
            "benchmark-tenant",
            "user-other",
            search_mode="dense",
        )

        assert len(benchmark_results) == 1
        assert other_tenant_results == []
        assert benchmark_results[0].payload["source"] == "benchmark"
        assert "filename" not in benchmark_results[0].payload
        assert "file_hash" not in benchmark_results[0].payload
        assert other_user_results == []
    finally:
        client.close()


def test_benchmark_reimport_preserves_user_upload(tmp_path, monkeypatch):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    vectors = np.zeros((1, 384), dtype=np.float32)
    vectors[0, 0] = 1.0
    record = {
        "chunk_id": "a" * 64,
        "document_id": "CVE-TEST-1",
        "position": 0,
        "text": "A test vulnerability record.",
        "chunker_version": "recursive-v1-size-380-overlap-50",
        "tenant_id": "benchmark-tenant",
        "user_id": "benchmark-user",
        "location": "description",
        "content_hash": "b" * 64,
    }
    monkeypatch.setattr(
        benchmark_data,
        "load_benchmark_artifacts",
        lambda directory: (vectors, [record]),
    )
    upload_chunk = Chunk(
        chunk_id="c" * 64,
        document_id="nist-upload",
        position=0,
        text="A user-uploaded NIST excerpt.",
        chunker_version="recursive-v1-size-380-overlap-50",
        tenant_id="benchmark-tenant",
        user_id="benchmark-user",
        location="page 1",
        content_hash="d" * 64,
    )

    try:
        vector_store.create_collection(client)
        upload_point = vector_store.point_for_chunk(
            upload_chunk,
            vectors[0].tolist(),
            source="user_upload",
            filename="nist.pdf",
            file_hash="e" * 64,
        )
        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=[upload_point],
            wait=True,
        )
        first_count = benchmark_data.import_benchmark_artifacts(client, tmp_path)
        second_count = benchmark_data.import_benchmark_artifacts(client, tmp_path)

        assert first_count == second_count == 1

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )
        user_uploads = [
            point for point in points if point.payload.get("source") == "user_upload"
        ]
        assert len(user_uploads) == 1
        assert user_uploads[0].payload["document_id"] == "nist-upload"
    finally:
        client.close()


LOCAL_BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark"


@pytest.mark.skipif(
    __import__("os").environ.get("RUN_BENCHMARK_INTEGRATION") != "1"
    or not (LOCAL_BENCHMARK_DIR / "manifest.json").is_file(),
    reason="Set RUN_BENCHMARK_INTEGRATION=1 and provide local benchmark files",
)
def test_full_benchmark_reimport_keeps_10792_and_upload(tmp_path):
    vectors, records = benchmark_data.load_benchmark_artifacts(LOCAL_BENCHMARK_DIR)
    assert len(records) == 10_792

    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        upload_chunk = Chunk(
            chunk_id="c" * 64,
            document_id="nist-upload",
            position=0,
            text="A user-uploaded NIST excerpt.",
            chunker_version="recursive-v1-size-380-overlap-50",
            tenant_id="benchmark-tenant",
            user_id="benchmark-user",
            location="page 1",
            content_hash="d" * 64,
        )
        upload_point = vector_store.point_for_chunk(
            upload_chunk,
            vectors[0].tolist(),
            source="user_upload",
            filename="nist.pdf",
            file_hash="e" * 64,
        )
        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=[upload_point],
            wait=True,
        )

        first_count = benchmark_data.import_benchmark_artifacts(
            client, LOCAL_BENCHMARK_DIR
        )
        second_count = benchmark_data.import_benchmark_artifacts(
            client, LOCAL_BENCHMARK_DIR
        )
        upload_count = client.count(
            collection_name=vector_store.COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value="benchmark-tenant"),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="user_upload"),
                    ),
                ]
            ),
            exact=True,
        )

        assert first_count == second_count == 10_792
        assert upload_count.count == 1
    finally:
        client.close()
