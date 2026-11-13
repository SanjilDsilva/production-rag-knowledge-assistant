from qdrant_client import QdrantClient

from production_rag.chunking import Chunk
from production_rag.embeddings import embed_documents, embed_query
from production_rag.vector_store import create_collection, search_chunks, upsert_chunks


def test_local_embedding_and_qdrant_search_find_sample_chunk(tmp_path):
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    try:
        create_collection(client)
        text = "Administrators must use multi-factor authentication."
        chunk = Chunk(
            chunk_id="sample-chunk-1",
            document_id="doc-1",
            position=0,
            text=text,
            chunker_version="test-v1",
            tenant_id="tenant-a",
            user_id="user-a",
            location="whole file",
            content_hash="c" * 64,
        )
        document_vector = embed_documents([text])[0]
        upsert_chunks(
            client,
            [chunk],
            [document_vector],
            filename="sample.txt",
            file_hash="f" * 64,
        )
        question = "What authentication do administrators need?"
        results = search_chunks(
            client=client,
            query_text=question,
            query_vector=embed_query(question),
            tenant_id="tenant-a",
            user_id="user-a",
            ready_document_ids=["doc-1"],
            limit=1,
        )
        assert results[0].payload["chunk_id"] == "sample-chunk-1"
    finally:
        client.close()
