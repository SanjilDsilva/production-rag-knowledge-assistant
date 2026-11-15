from production_rag import embeddings
from production_rag.answer_cache import SQLiteCacheBackend
import logging
import pytest


class FakeModel:
    def __init__(self):
        self.received_texts = []
        self.normalize_flags = []

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        self.received_texts.append(texts)
        self.normalize_flags.append(normalize_embeddings)
        return [[1.0, 0.0] for _ in texts]


def test_query_gets_instruction_but_document_text_does_not(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(
        embeddings,
        "get_embedding_model",
        lambda: fake_model,
    )

    query_vector = embeddings.embed_query("What is the budget?")
    document_vectors = embeddings.embed_documents(["The budget is 500."])

    assert fake_model.received_texts == [
        [
            "Represent this sentence for searching relevant passages: what is the budget?"
        ],
        ["The budget is 500."],
    ]
    assert fake_model.normalize_flags == [True, True]
    assert query_vector == [1.0, 0.0]
    assert document_vectors == [[1.0, 0.0]]


def test_query_embedding_cache_reuses_and_evicts_oldest(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: fake_model)
    test_cache = embeddings.BoundedAnswerCache[list[float]](
        max_entries=2,
        ttl_seconds=30,
    )
    monkeypatch.setattr(embeddings, "_query_cache", test_cache)

    first_a = embeddings.embed_query("question a")
    first_a[0] = 9.0
    embeddings.embed_query("question b")
    repeated_a = embeddings.embed_query("question a")
    embeddings.embed_query("question c")
    reloaded_b = embeddings.embed_query("question b")

    assert repeated_a == [1.0, 0.0]
    assert reloaded_b == [1.0, 0.0]
    assert len(fake_model.received_texts) == 4
    assert test_cache.stats()["hits"] == 1
    assert test_cache.stats()["misses"] == 4


def test_query_embedding_cache_key_tracks_query_and_model_settings(monkeypatch):
    fake_model = FakeModel()
    original_prefix = embeddings.QUERY_PREFIX
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: fake_model)
    test_cache = embeddings.BoundedAnswerCache[list[float]]()
    monkeypatch.setattr(embeddings, "_query_cache", test_cache)

    embeddings.embed_query("  WHAT   is this?  ")
    embeddings.embed_query("what is this?")

    monkeypatch.setattr(embeddings, "MODEL_REVISION", "another-revision")
    embeddings.embed_query("what is this?")

    monkeypatch.setattr(embeddings, "QUERY_PREFIX", "Custom instruction: ")
    embeddings.embed_query("what is this?")

    assert fake_model.received_texts == [
        [f"{original_prefix}what is this?"],
        [f"{original_prefix}what is this?"],
        ["Custom instruction: what is this?"],
    ]
    assert test_cache.stats()["hits"] == 1
    assert test_cache.stats()["misses"] == 3


def test_query_cache_factory_selects_memory_and_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("RAG_CACHE_BACKEND", raising=False)

    memory_cache = embeddings.create_query_cache()
    assert isinstance(memory_cache, embeddings.BoundedAnswerCache)

    monkeypatch.setenv("RAG_CACHE_BACKEND", "sqlite")
    sqlite_cache = embeddings.create_query_cache()

    assert isinstance(sqlite_cache, SQLiteCacheBackend)
    assert sqlite_cache.namespace == "query_embeddings"
    assert sqlite_cache.database_path == tmp_path / "cache.db"


def test_query_embedding_sqlite_cache_survives_backend_recreation(
    tmp_path, monkeypatch
):
    fake_model = FakeModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: fake_model)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CACHE_BACKEND", "sqlite")

    first_cache = embeddings.create_query_cache()
    monkeypatch.setattr(embeddings, "_query_cache", first_cache)
    first_vector = embeddings.embed_query("What is the answer?")

    second_cache = embeddings.create_query_cache()
    monkeypatch.setattr(embeddings, "_query_cache", second_cache)
    second_vector = embeddings.embed_query("What is the answer?")

    assert first_vector == second_vector
    assert len(fake_model.received_texts) == 1
    assert second_cache.stats()["hits"] == 1


@pytest.mark.parametrize("backend_name", ["memory", "sqlite"])
def test_query_embedding_cache_logs_backend_stats(
    tmp_path, monkeypatch, caplog, backend_name
):
    fake_model = FakeModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: fake_model)
    monkeypatch.setenv("RAG_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("RAG_CACHE_BACKEND", backend_name)
    monkeypatch.setattr(embeddings, "_query_cache", embeddings.create_query_cache())
    caplog.set_level(logging.INFO, logger="production_rag.embeddings")

    embeddings.embed_query("What is the answer?")
    embeddings.embed_query("What is the answer?")

    records = [
        record
        for record in caplog.records
        if record.message == "query_embedding_cache_lookup"
    ]

    assert [record.cache_hit for record in records] == [False, True]
    assert all(record.cache_backend == backend_name for record in records)
    assert records[-1].cache_hits == 1
    assert records[-1].cache_misses == 1
