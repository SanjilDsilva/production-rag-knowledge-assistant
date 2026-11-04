from qdrant_client import QdrantClient, models
from uuid import NAMESPACE_URL, uuid5
from production_rag.chunking import Chunk


COLLECTION_NAME = "knowledge_chunks"
TEXT_FIELD = "text"
DENSE_VECTOR_NAME = "dense"
DENSE_VECTOR_SIZE = 384
POINT_ID_NAMESPACE = NAMESPACE_URL


def point_id_for_chunk_id(chunk_id: str) -> str:
    name = f"production-rag:chunk:{chunk_id}"
    return str(uuid5(POINT_ID_NAMESPACE, name))


def find_document_by_file_hash(
    client: QdrantClient,
    tenant_id: str,
    user_id: str,
    file_hash: str,
) -> str | None:
    matches, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchValue(value=file_hash),
                ),
            ]
        ),
        limit=1,
        with_payload=["document_id"],
        with_vectors=False,
    )

    if not matches:
        return None
    return matches[0].payload.get("document_id")


def find_document_by_filename(
    client: QdrantClient,
    tenant_id: str,
    user_id: str,
    filename: str,
) -> str | None:
    matches, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="tenant_id",
                    match=models.MatchValue(value=tenant_id),
                ),
                models.FieldCondition(
                    key="user_id",
                    match=models.MatchValue(value=user_id),
                ),
                models.FieldCondition(
                    key="filename_key",
                    match=models.MatchValue(value=filename.casefold()),
                ),
            ]
        ),
        limit=1,
        with_payload=["document_id"],
        with_vectors=False,
    )
    if not matches:
        return None
    return matches[0].payload.get("document_id")


def delete_document_chunks(
    client: QdrantClient,
    tenant_id: str,
    user_id: str,
    document_id: str,
) -> None:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value=user_id),
                    ),
                    models.FieldCondition(
                        key="document_id",
                        match=models.MatchValue(value=document_id),
                    ),
                ]
            )
        ),
        wait=True,
    )


def upsert_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    vectors: list[list[float]],
    filename: str,
    file_hash: str,
) -> None:
    if len(chunks) != len(vectors):
        raise ValueError("Each chunk must have exactly one vector")

    points = []
    for chunk, vector in zip(chunks, vectors):
        point = point_for_chunk(
            chunk,
            vector,
            source="user_upload",
            filename=filename,
            file_hash=file_hash,
        )
        points.append(point)

    if points:
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )


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

    for field_name in ("user_id", "document_id", "file_hash", "filename_key", "source"):
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
    search_mode: str = "hybrid",
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

    if search_mode not in ("dense", "keyword", "hybrid"):
        raise ValueError("Search_mode must be dense, keyword, or hybrid")

    if search_mode == "dense":
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            using=DENSE_VECTOR_NAME,
            query_filter=access_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

    if search_mode == "keyword":
        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=models.TextQuery(text=query_text),
            using=TEXT_FIELD,
            query_filter=access_filter,
            limit=limit,
            with_payload=True,
        )
        return response.points

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


def point_for_chunk(
    chunk: Chunk,
    vector: list[float],
    *,
    source: str,
    filename: str | None = None,
    file_hash: str | None = None,
) -> models.PointStruct:
    if len(vector) != DENSE_VECTOR_SIZE:
        raise ValueError("Vector has the wrong size")
    if source not in ("user_upload", "benchmark"):
        raise ValueError("Unknown point source")

    payload = {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "position": chunk.position,
        "text": chunk.text,
        "chunker_version": chunk.chunker_version,
        "tenant_id": chunk.tenant_id,
        "user_id": chunk.user_id,
        "location": chunk.location,
        "content_hash": chunk.content_hash,
        "source": source,
    }

    if source == "user_upload":
        if filename is None or file_hash is None:
            raise ValueError("Uploads need a filename and file hash")
        payload["filename"] = filename
        payload["filename_key"] = filename.casefold()
        payload["file_hash"] = file_hash
    elif filename is not None or file_hash is not None:
        raise ValueError("Benchmark points must not have file-specific fields")

    return models.PointStruct(
        id=point_id_for_chunk_id(chunk.chunk_id),
        vector={DENSE_VECTOR_NAME: vector},
        payload=payload,
    )
