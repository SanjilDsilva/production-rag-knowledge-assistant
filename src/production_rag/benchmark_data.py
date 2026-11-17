import hashlib
from pathlib import Path
import json
from typing import Any
import numpy as np
from production_rag.chunking import Chunk
import time


EXPECTED_MODEL = "BAAI/bge-small-en-v1.5"
EXPECTED_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
EXPECTED_CHUNKER = "recursive-v1-size-380-overlap-50"


def _chunk_from_record(record: dict[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=record["chunk_id"],
        document_id=record["document_id"],
        position=record["position"],
        text=record["text"],
        chunker_version=record["chunker_version"],
        tenant_id=record["tenant_id"],
        user_id=record["user_id"],
        location=record["location"],
        content_hash=record["content_hash"],
    )


def sha256_file(path: Path) -> str:
    if path.suffix == ".jsonl":
        text = path.read_text(encoding="utf-8")
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)

    return digest.hexdigest()


def verify_benchmark_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    file_hashes = manifest.get("files")

    if not isinstance(file_hashes, dict):
        raise ValueError("Manifest is missing its file checksums")

    for filename in ("vectors.npy", "chunks.jsonl"):
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"Benchmark file not found: {path}")

        expected_hash = file_hashes.get(filename)
        if not isinstance(expected_hash, str):
            raise ValueError(f"Manifest has no checksum for {filename}")

        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise ValueError(f"Checksum mismatch for {filename}")

    return manifest


def load_benchmark_artifacts(
    directory: Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    manifest = verify_benchmark_manifest(directory)

    if manifest.get("embedding_model") != EXPECTED_MODEL:
        raise ValueError("Unexpected embedding model")
    if manifest.get("embedding_model_revision") != EXPECTED_REVISION:
        raise ValueError("Unexpected embedding model revision")
    if manifest.get("chunker_version") != EXPECTED_CHUNKER:
        raise ValueError("Unexpected chunker version")
    if manifest.get("vectors_normalized") is not True:
        raise ValueError("Manifest says vectors are not normalized")

    vectors = np.load(directory / "vectors.npy", allow_pickle=False)
    if vectors.dtype != np.float32:
        raise ValueError("Vectors must use float32")
    expected_shape = (manifest.get("chunk_count"), manifest.get("vector_size"))
    if vectors.shape != expected_shape:
        raise ValueError("Vector shape does not match the manifest")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5):
        raise ValueError("Vectors are not normalized")

    chunks = []
    with (directory / "chunks.jsonl").open(encoding="utf-8") as file:
        for line in file:
            chunk = json.loads(line)
            if chunk.get("tenant_id") != "benchmark-tenant":
                raise ValueError("Unexpected benchmark tenant")
            if chunk.get("user_id") != "benchmark-user":
                raise ValueError("Unexpected benchmark user")
            if chunk.get("chunker_version") != EXPECTED_CHUNKER:
                raise ValueError("Chunk version does not match the manifest")
            chunks.append(chunk)

    if len(chunks) != len(vectors):
        raise ValueError("Vector and metadata counts do not match")

    return vectors, chunks


from qdrant_client import QdrantClient, models
from production_rag.vector_store import (
    COLLECTION_NAME,
    create_collection,
    point_for_chunk,
)


def import_benchmark_artifacts(
    client: QdrantClient,
    directory: Path,
    batch_size: int = 256,
) -> int:
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1")

    started = time.perf_counter()
    vectors, records = load_benchmark_artifacts(directory)
    create_collection(client)

    total = len(records)
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        points = []

        for record, vector in zip(records[start:stop], vectors[start:stop]):
            chunk = _chunk_from_record(record)
            points.append(
                point_for_chunk(
                    chunk,
                    vector.tolist(),
                    source="benchmark",
                )
            )
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )
        print(f"Imported {stop}/{total} benchmark chunks")

    count_result = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value="benchmark-tenant"),
                ),
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value="benchmark"),
                ),
            ]
        ),
        exact=True,
    )

    if count_result.count != total:
        raise RuntimeError(
            f"Expected {total} benchmark points, found {count_result.count}"
        )

    elapsed = time.perf_counter() - started
    print(f"Import complete: {count_result.count} points in {elapsed:.2f}s")
    return count_result.count
