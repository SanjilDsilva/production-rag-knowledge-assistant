import production_rag.answer_cache as answer_cache
import sqlite3
from contextlib import closing


def test_normalize_question_ignores_case_and_extra_spaces():
    assert answer_cache.normalize_question("  HOW   are You? \n") == "how are you?"


def test_answer_cache_key_uses_identity_and_versions():
    def make_key(
        tenant="tenant-a",
        user="user-a",
        document_version="docs-v1",
        prompt_version="prompt-v1",
        model="gemini-model",
    ):
        return answer_cache.make_answer_cache_key(
            tenant,
            user,
            "What is the policy?",
            document_version,
            prompt_version,
            model,
        )

    key = make_key()

    assert key == make_key()
    assert key != make_key(tenant="tenant-b")
    assert key != make_key(user="user-b")
    assert key != make_key(document_version="docs-v2")
    assert key != make_key(prompt_version="prompt-v2")
    assert key != make_key(model="different-model")


def test_answer_cache_evicts_least_recently_used_entry():
    cache = answer_cache.BoundedAnswerCache(max_entries=2, ttl_seconds=30)
    cache.set("a", "answer-a")
    cache.set("b", "answer-b")

    assert cache.get("a") == "answer-a"

    cache.set("c", "answer-c")

    assert cache.get("b") is None
    assert cache.get("a") == "answer-a"
    assert cache.get("c") == "answer-c"


def test_answer_cache_expires_entry(monkeypatch):
    current_time = [100.0]
    monkeypatch.setattr(
        "production_rag.answer_cache.monotonic",
        lambda: current_time[0],
    )
    cache = answer_cache.BoundedAnswerCache(max_entries=2, ttl_seconds=5)
    cache.set("a", "answer-a")

    current_time[0] = 104.0
    assert cache.get("a") == "answer-a"

    current_time[0] = 105.0
    assert cache.get("a") is None


def test_answer_cache_tracks_hits_and_misses():
    cache = answer_cache.BoundedAnswerCache(max_entries=2, ttl_seconds=30)

    cache.set("a", "answer-a")
    assert cache.get("a") == "answer-a"
    assert cache.get("missing") is None

    assert cache.stats() == {
        "hits": 1,
        "misses": 1,
        "hit_rate": 0.5,
    }


def test_answer_cache_key_uses_visible_document_set():
    def make_key(document_ids):
        return answer_cache.make_answer_cache_key(
            "tenant-a",
            "user-a",
            "What is the policy?",
            "docs-v1",
            "prompt-v1",
            "gemini-model",
            visible_document_ids=document_ids,
        )

    assert make_key(["doc-a", "doc-b"]) == make_key(["doc-b", "doc-a"])
    assert make_key(["doc-a", "doc-b"]) != make_key(["doc-a"])


def test_answer_cache_can_delete_one_entry_or_clear_all():
    cache = answer_cache.BoundedAnswerCache()

    cache.set("a", "answer-a")
    cache.set("b", "answer-b")
    cache.delete("a")

    assert cache.get("a") is None
    assert cache.get("b") == "answer-b"

    cache.clear()

    assert cache.get("b") is None


def test_bounded_cache_can_store_embedding_vector():
    cache: answer_cache.BoundedAnswerCache[list[float]] = (
        answer_cache.BoundedAnswerCache()
    )
    vector = [0.25, 0.75]

    cache.set("embedding-key", vector)

    assert cache.get("embedding-key") == vector


def test_sqlite_cache_initializes_wal_database(tmp_path):
    cache = answer_cache.SQLiteCacheBackend(tmp_path / "cache.db")

    with closing(sqlite3.connect(cache.database_path)) as connection:
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(cache_entries)")
        }

    assert journal_mode.lower() == "wal"
    assert {
        "namespace",
        "cache_key",
        "cache_value",
        "expires_at",
        "last_accessed",
    } <= columns


def test_sqlite_cache_deletes_expired_entries(tmp_path):
    cache = answer_cache.SQLiteCacheBackend(tmp_path / "cache.db")

    with closing(sqlite3.connect(cache.database_path)) as connection:
        connection.execute(
            """
            INSERT INTO cache_entries
                (namespace, cache_key, cache_value, expires_at, last_accessed)
            VALUES (?, ?, ?, ?, ?)
            """,
            ("answers", "old-key", '"old answer"', 0.0, 0.0),
        )
        connection.commit()

    assert cache.get("old-key") is None

    with closing(sqlite3.connect(cache.database_path)) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM cache_entries WHERE cache_key = ?",
            ("old-key",),
        ).fetchone()[0]

    assert count == 0


def test_sqlite_cache_stores_values_and_enforces_entry_limit(tmp_path):
    database_path = tmp_path / "cache.db"
    cache = answer_cache.SQLiteCacheBackend[str](database_path, max_entries=2)

    cache.set("a", "answer-a")
    cache.set("b", "answer-b")
    assert cache.get("a") == "answer-a"
    cache.set("c", "answer-c")

    assert cache.get("a") == "answer-a"
    assert cache.get("b") is None
    assert cache.get("c") == "answer-c"

    vector_cache = answer_cache.SQLiteCacheBackend[list[float]](
        database_path, namespace="embeddings"
    )
    vector_cache.set("question-key", [0.25, 0.75])

    assert vector_cache.get("question-key") == [0.25, 0.75]


def test_sqlite_cache_delete_clear_and_stats_are_namespace_scoped(tmp_path):
    database_path = tmp_path / "cache.db"
    answers = answer_cache.SQLiteCacheBackend[str](database_path)
    embeddings = answer_cache.SQLiteCacheBackend[list[float]](
        database_path, namespace="embeddings"
    )

    answers.set("same-key", "answer")
    embeddings.set("same-key", [0.25, 0.75])
    answers.delete("same-key")

    assert answers.get("same-key") is None
    assert embeddings.get("same-key") == [0.25, 0.75]

    answers.set("clear-me", "answer")
    answers.clear()

    assert answers.get("clear-me") is None
    assert answers.stats() == {"hits": 0, "misses": 2, "hit_rate": 0.0}
    assert embeddings.stats() == {"hits": 1, "misses": 0, "hit_rate": 1.0}


def test_cache_backend_factory_creates_memory_and_sqlite(tmp_path):
    memory_cache = answer_cache.create_cache_backend(
        "memory",
        namespace="answers",
        max_entries=4,
        ttl_seconds=30,
    )
    sqlite_cache = answer_cache.create_cache_backend(
        "sqlite",
        namespace="answers",
        max_entries=4,
        ttl_seconds=30,
        database_path=tmp_path / "cache.db",
    )

    assert isinstance(memory_cache, answer_cache.BoundedAnswerCache)
    assert isinstance(sqlite_cache, answer_cache.SQLiteCacheBackend)
    assert memory_cache.backend_name == "memory"
    assert sqlite_cache.backend_name == "sqlite"
