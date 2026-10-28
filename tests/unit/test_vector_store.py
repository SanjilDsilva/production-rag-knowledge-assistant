from production_rag import vector_store
from qdrant_client import QdrantClient, models


def test_search_returns_only_requesting_tenant_and_user(tmp_path):
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
            tenant_id="tenant-a",
            user_id="user-a",
            limit=10,
        )

        returned_document_ids = {
            result.payload["document_id"] for result in results
        }
        assert returned_document_ids == {"doc-a"}
    finally:
        client.close()