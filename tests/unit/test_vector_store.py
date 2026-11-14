from production_rag import vector_store
from qdrant_client import QdrantClient, models
from uuid import UUID
import pytest
from dataclasses import replace
from production_rag.chunking import Chunk


@pytest.mark.parametrize("search_mode", ["dense", "keyword", "hybrid"])
@pytest.mark.parametrize(
    ("tenant_id", "user_id", "expected_document_id"),
    [
        ("tenant-a", "user-a", "doc-a"),
        ("tenant-a", "user-b", "doc-b"),
        ("tenant-b", "user-a", "doc-c"),
    ],
)
def test_search_returns_only_requesting_tenant_and_user(
    tmp_path, search_mode, tenant_id, user_id, expected_document_id
):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        dense_vector = [1.0] + [0.0] * 383

        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,
                    vector={vector_store.DENSE_VECTOR_NAME: dense_vector},
                    payload={
                        "tenant_id": "tenant-a",
                        "user_id": "user-a",
                        "document_id": "doc-a",
                        "text": "Quarterly budget is 500 dollars.",
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector={vector_store.DENSE_VECTOR_NAME: dense_vector},
                    payload={
                        "tenant_id": "tenant-a",
                        "user_id": "user-b",
                        "document_id": "doc-b",
                        "text": "Quarterly budget is 500 dollars.",
                    },
                ),
                models.PointStruct(
                    id=3,
                    vector={vector_store.DENSE_VECTOR_NAME: dense_vector},
                    payload={
                        "tenant_id": "tenant-b",
                        "user_id": "user-a",
                        "document_id": "doc-c",
                        "text": "Quarterly budget is 500 dollars.",
                    },
                ),
            ],
        )

        results = vector_store.search_chunks(
            client=client,
            query_text="quarterly budget",
            query_vector=dense_vector,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=10,
            search_mode=search_mode,
        )

        returned_document_ids = {result.payload["document_id"] for result in results}
        assert returned_document_ids == {expected_document_id}
    finally:
        client.close()


def test_qdrant_point_id_is_stable_and_differs_for_different_chunk_ids():
    chunk_id_a = "a" * 64
    chunk_id_b = "b" * 64

    first_id = vector_store.point_id_for_chunk_id(chunk_id_a)
    repeated_id = vector_store.point_id_for_chunk_id(chunk_id_a)
    second_id = vector_store.point_id_for_chunk_id(chunk_id_b)

    assert first_id == repeated_id
    assert first_id != second_id
    UUID(first_id)
    UUID(second_id)


def test_same_text_from_two_tenants_is_stored_as_two_qdrant_points(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        tenant_a_chunk = Chunk(
            chunk_id="a" * 64,
            document_id="same-document",
            position=0,
            text="The same useful text belongs to both tenants.",
            chunker_version="recursive-v1",
            tenant_id="tenant-a",
            user_id="same-user",
            location="page 1",
            content_hash="c" * 64,
        )
        tenant_b_chunk = replace(
            tenant_a_chunk,
            chunk_id="b" * 64,
            tenant_id="tenant-b",
        )
        dense_vector = [1.0] + [0.0] * 383

        vector_store.upsert_chunks(
            client,
            [tenant_a_chunk, tenant_b_chunk],
            [dense_vector, dense_vector],
            filename="shared.txt",
            file_hash="f" * 64,
        )

        vector_store.upsert_chunks(
            client,
            [tenant_a_chunk, tenant_b_chunk],
            [dense_vector, dense_vector],
            filename="shared.txt",
            file_hash="f" * 64,
        )

        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
        )
        assert {point.payload["filename"] for point in points} == {"shared.txt"}
        assert {point.payload["file_hash"] for point in points} == {"f" * 64}
        assert len(points) == 2
        assert {point.payload["tenant_id"] for point in points} == {
            "tenant-a",
            "tenant-b",
        }
        assert {point.payload["chunk_id"] for point in points} == {
            tenant_a_chunk.chunk_id,
            tenant_b_chunk.chunk_id,
        }
    finally:
        client.close()


def test_file_hash_lookup_is_scoped_to_tenant_and_user(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        dense_vector = [1.0] + [0.0] * 383
        file_hash = "f" * 64
        owners = [
            (1, "tenant-a", "user-a", "doc-a"),
            (2, "tenant-a", "user-b", "doc-b"),
            (3, "tenant-b", "user-a", "doc-c"),
        ]
        points = []

        for point_id, tenant_id, user_id, document_id in owners:
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={vector_store.DENSE_VECTOR_NAME: dense_vector},
                    payload={
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "document_id": document_id,
                        "file_hash": file_hash,
                        "filename_key": "shared.txt",
                    },
                )
            )

        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=points,
        )

        assert (
            vector_store.find_document_by_file_hash(
                client, "tenant-a", "user-a", file_hash
            )
            == "doc-a"
        )
        assert (
            vector_store.find_document_by_file_hash(
                client, "tenant-a", "user-b", file_hash
            )
            == "doc-b"
        )
        assert (
            vector_store.find_document_by_file_hash(
                client, "tenant-b", "user-a", file_hash
            )
            == "doc-c"
        )
        assert (
            vector_store.find_document_by_file_hash(
                client, "tenant-b", "user-b", file_hash
            )
            is None
        )
        assert (
            vector_store.find_document_by_filename(
                client, "tenant-a", "user-a", "shared.txt"
            )
            == "doc-a"
        )
        assert (
            vector_store.find_document_by_filename(
                client, "tenant-a", "user-b", "shared.txt"
            )
            == "doc-b"
        )

        vector_store.delete_document_chunks(client, "tenant-a", "user-a", "doc-a")
        remaining_points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
        )
        assert {point.payload["document_id"] for point in remaining_points} == {
            "doc-b",
            "doc-c",
        }
    finally:
        client.close()


def test_upload_upsert_marks_source_as_user_upload(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        chunk = Chunk(
            chunk_id="a" * 64,
            document_id="nist-upload",
            position=0,
            text="A user-uploaded NIST excerpt.",
            chunker_version="recursive-v1",
            tenant_id="benchmark-tenant",
            user_id="benchmark-user",
            location="page 1",
            content_hash="b" * 64,
        )
        dense_vector = [1.0] + [0.0] * 383

        vector_store.upsert_chunks(
            client,
            [chunk],
            [dense_vector],
            filename="nist.pdf",
            file_hash="c" * 64,
        )
        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=10,
            with_payload=True,
            with_vectors=False,
        )

        assert len(points) == 1
        assert points[0].payload["source"] == "user_upload"
    finally:
        client.close()


@pytest.mark.parametrize("search_mode", ["dense", "keyword", "hybrid"])
def test_search_returns_only_ready_documents(tmp_path, search_mode):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        vector = [1.0] + [0.0] * 383

        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=[
                models.PointStruct(
                    id=1,
                    vector={vector_store.DENSE_VECTOR_NAME: vector},
                    payload={
                        "tenant_id": "tenant-a",
                        "user_id": "user-a",
                        "document_id": "ready-doc",
                        "text": "Project policy is ready.",
                    },
                ),
                models.PointStruct(
                    id=2,
                    vector={vector_store.DENSE_VECTOR_NAME: vector},
                    payload={
                        "tenant_id": "tenant-a",
                        "user_id": "user-a",
                        "document_id": "processing-doc",
                        "text": "Project policy is still processing.",
                    },
                ),
            ],
        )

        results = vector_store.search_chunks(
            client=client,
            query_text="project policy",
            query_vector=vector,
            tenant_id="tenant-a",
            user_id="user-a",
            ready_document_ids=["ready-doc"],
            search_mode=search_mode,
        )

        assert {point.payload["document_id"] for point in results} == {"ready-doc"}

        no_ready_results = vector_store.search_chunks(
            client=client,
            query_text="project policy",
            query_vector=vector,
            tenant_id="tenant-a",
            user_id="user-a",
            ready_document_ids=[],
            search_mode=search_mode,
        )
        assert no_ready_results == []
    finally:
        client.close()


@pytest.mark.parametrize("visibility", ["private", "tenant_shared"])
def test_user_upload_point_stores_visibility(visibility):
    chunk = Chunk(
        chunk_id="a" * 64,
        document_id="doc-a",
        position=0,
        text="Example upload text.",
        chunker_version="test-v1",
        tenant_id="tenant-a",
        user_id="user-a",
        location="page 1",
        content_hash="b" * 64,
    )

    point = vector_store.point_for_chunk(
        chunk,
        [1.0] + [0.0] * 383,
        source="user_upload",
        filename="example.txt",
        file_hash="c" * 64,
        visibility=visibility,
    )

    assert point.payload["visibility"] == visibility


def test_upsert_chunks_stores_shared_visibility(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        chunk = Chunk(
            chunk_id="d" * 64,
            document_id="shared-doc",
            position=0,
            text="Shared upload text.",
            chunker_version="test-v1",
            tenant_id="tenant-a",
            user_id="user-a",
            location="page 1",
            content_hash="e" * 64,
        )

        vector_store.upsert_chunks(
            client,
            [chunk],
            [[1.0] + [0.0] * 383],
            filename="shared.txt",
            file_hash="f" * 64,
            visibility="tenant_shared",
        )
        points, _ = client.scroll(
            collection_name=vector_store.COLLECTION_NAME,
            limit=1,
            with_payload=True,
        )

        assert points[0].payload["visibility"] == "tenant_shared"
    finally:
        client.close()


@pytest.mark.parametrize("search_mode", ["dense", "keyword", "hybrid"])
@pytest.mark.parametrize(
    ("role", "expected_ids"),
    [
        ("member", {"own-private", "shared"}),
        ("admin", {"own-private", "shared", "other-private"}),
    ],
)
def test_search_chunks_enforces_role_visibility_and_tenant(
    tmp_path, search_mode, role, expected_ids
):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        vector_store.create_collection(client)
        vector = [1.0] + [0.0] * 383
        records = [
            ("tenant-a", "user-a", "own-private", "private"),
            ("tenant-a", "user-b", "shared", "tenant_shared"),
            ("tenant-a", "user-b", "other-private", "private"),
            ("tenant-b", "user-b", "foreign-shared", "tenant_shared"),
        ]
        points = [
            models.PointStruct(
                id=index,
                vector={vector_store.DENSE_VECTOR_NAME: vector},
                payload={
                    "tenant_id": tenant,
                    "user_id": owner,
                    "document_id": document_id,
                    "visibility": visibility,
                    "text": "shared access control",
                },
            )
            for index, (tenant, owner, document_id, visibility) in enumerate(
                records, start=1
            )
        ]
        client.upsert(
            collection_name=vector_store.COLLECTION_NAME,
            points=points,
            wait=True,
        )

        results = vector_store.search_chunks(
            client=client,
            query_text="shared access",
            query_vector=vector,
            tenant_id="tenant-a",
            user_id="user-a",
            role=role,
            limit=10,
            search_mode=search_mode,
        )

        assert {point.payload["document_id"] for point in results} == expected_ids
    finally:
        client.close()
