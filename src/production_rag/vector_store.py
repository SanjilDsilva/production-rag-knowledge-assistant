from qdrant_client import QdrantClient, models

COLLECTION_NAME = "knowledge_chunks"
TEXT_FIELD = "text"
DENSE_VECTOR_NAME = "dense"
DENSE_VECTOR_SIZE = 384


def create_collection(client: QdrantClient) -> None:
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=models.Distance.COSINE,
                )
            },
        )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="tenant_id",
        field_schema=models.KeywordIndexParams(
            type=models.KeywordIndexType.KEYWORD,
            is_tenant=True,
        ),
    )

    for field_name in ("user_id", "document_id"):
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name=TEXT_FIELD,
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            scoring=models.TextScoringParams(
                type=models.TextScoringType.BM25,
            ),
        ),
    )


def search_chunks(
    client: QdrantClient,
    query_text: str,
    query_vector: list[float],
    tenant_id: str,
    user_id: str,
    limit: int = 5,
) -> list[models.ScoredPoint]:
    access_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="tenant_id",
                match=models.MatchValue(value=tenant_id),
            ),
            models.FieldCondition(
                key="user_id",
                match=models.MatchValue(value=user_id),
            ),
        ]
    )

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=DENSE_VECTOR_NAME,
                filter=access_filter,
                limit=limit,
            ),
            models.Prefetch(
                query=models.TextQuery(text=query_text),
                using=TEXT_FIELD,
                filter=access_filter,
                limit=limit,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf()),
        query_filter=access_filter,
        limit=limit,
        with_payload=True,
    )
    return response.points
