import hashlib
import json
from collections import OrderedDict
from threading import Lock
from time import monotonic
from typing import Protocol, TypeVar, Generic
import sqlite3
from contextlib import closing
from pathlib import Path
from time import monotonic, time as wall_time


CacheValue = TypeVar("CacheValue")


class CacheBackend(Protocol[CacheValue]):
    """Operations that each cache implementation must support."""

    backend_name: str

    def get(self, key: str) -> CacheValue | None: ...

    def set(self, key: str, value: CacheValue) -> None: ...

    def delete(self, key: str) -> None: ...

    def clear(self) -> None: ...

    def stats(self) -> dict[str, int | float]: ...


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def make_answer_cache_key(
    tenant_id: str,
    user_id: str,
    question: str,
    document_set_version: str,
    prompt_version: str,
    model_name: str,
    visible_document_ids: list[str] | None = None,
) -> str:
    visible_ids = sorted(set(visible_document_ids or []))
    visible_ids_json = json.dumps(
        visible_ids,
        separators=(",", ":"),
    ).encode("utf-8")
    visible_ids_hash = hashlib.sha256(visible_ids_json).hexdigest()

    key_parts = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "question": normalize_question(question),
        "document_set_version": document_set_version,
        "visible_document_ids_hash": visible_ids_hash,
        "prompt_version": prompt_version,
        "model_name": model_name,
    }

    encoded_key = json.dumps(
        key_parts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_key).hexdigest()


class BoundedAnswerCache(Generic[CacheValue]):
    backend_name = "memory"
    def __init__(
        self,
        max_entries: int = 256,
        ttl_seconds: int = 900,
    ) -> None:
        if max_entries < 1:
            raise ValueError("Cache must allow at least one entry")
        if ttl_seconds < 1:
            raise ValueError("Cache time limit must be positive")

        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, tuple[CacheValue, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> CacheValue | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None

            answer, expires_at = entry
            if expires_at <= monotonic():
                del self._entries[key]
                self._misses += 1
                return None

            self._entries.move_to_end(key)
            self._hits += 1
            return answer

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }

    def set(self, key: str, value: CacheValue) -> None:
        with self._lock:
            expires_at = monotonic() + self.ttl_seconds
            self._entries[key] = (value, expires_at)
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


class SQLiteCacheBackend(Generic[CacheValue]):
    backend_name = "sqlite"
    def __init__(
        self,
        database_path: Path,
        namespace: str = "answers",
        max_entries: int = 256,
        ttl_seconds: int = 900,
    ) -> None:
        if max_entries < 1 or ttl_seconds < 1:
            raise ValueError("Cache limits must be positive")

        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.namespace = namespace
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._hits = 0
        self._misses = 0
        self._metrics_lock = Lock()

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA busy_timeout=5000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                cache_value TEXT NOT NULL,
                expires_at REAL NOT NULL,
                last_accessed REAL NOT NULL,
                PRIMARY KEY (namespace, cache_key)
                )
                """
            )
            connection.commit()

    def get(self, key: str) -> CacheValue | None:
        now = wall_time()
        with closing(sqlite3.connect(self.database_path, timeout=5.0)) as connection:
            row = connection.execute(
                """
                SELECT cache_value, expires_at
                FROM cache_entries
                WHERE namespace = ? AND cache_key = ?
                """,
                (self.namespace, key),
            ).fetchone()

            if row is None:
                with self._metrics_lock:
                    self._misses += 1
                return None

            cache_value, expires_at = row
            if expires_at <= now:
                connection.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE namespace = ? AND cache_key = ?
                    """,
                    (self.namespace, key),
                )
                connection.commit()
                with self._metrics_lock:
                    self._misses += 1
                return None
            connection.execute(
                """
                UPDATE cache_entries
                SET last_accessed = ?
                WHERE namespace = ? AND cache_key = ?
                """,
                (now, self.namespace, key),
            )
            connection.commit()

        with self._metrics_lock:
            self._hits += 1
        return json.loads(cache_value)

    def set(self, key: str, value: CacheValue) -> None:
        now = wall_time()
        expires_at = now + self.ttl_seconds
        encoded_value = json.dumps(value, separators=(",", ":"))

        with closing(sqlite3.connect(self.database_path, timeout=5.0)) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND expires_at <= ?",
                    (self.namespace, now),
                )
                connection.execute(
                    """
                    INSERT INTO cache_entries
                        (namespace, cache_key, cache_value, expires_at, last_accessed)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(namespace, cache_key) DO UPDATE SET
                        cache_value = excluded.cache_value,
                        expires_at = excluded.expires_at,
                        last_accessed = excluded.last_accessed
                    """,
                    (self.namespace, key, encoded_value, expires_at, now),
                )
                connection.execute(
                    """
                    DELETE FROM cache_entries
                    WHERE namespace = ?
                    AND cache_key NOT IN (
                        SELECT cache_key
                        FROM cache_entries
                        WHERE namespace = ?
                        ORDER BY last_accessed DESC, cache_key DESC
                        LIMIT ?
                    )
                    """,
                    (self.namespace, self.namespace, self.max_entries),
                )

    def delete(self, key: str) -> None:
        with closing(sqlite3.connect(self.database_path, timeout=5.0)) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ? AND cache_key = ?",
                    (self.namespace, key),
                )

    def clear(self) -> None:
        with closing(sqlite3.connect(self.database_path, timeout=5.0)) as connection:
            with connection:
                connection.execute(
                    "DELETE FROM cache_entries WHERE namespace = ?",
                    (self.namespace,),
                )

    def stats(self) -> dict[str, int | float]:
        with self._metrics_lock:
            total = self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


def create_cache_backend(
    backend_name: str,
    *,
    namespace: str,
    max_entries: int,
    ttl_seconds: int,
    database_path: Path | None = None,
) -> CacheBackend[CacheValue]:
    backend_name = backend_name.casefold()

    if backend_name == "memory":
        return BoundedAnswerCache(
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )

    if backend_name == "sqlite":
        if database_path is None:
            raise ValueError("SQLite cache needs a database path")
        return SQLiteCacheBackend(
            database_path,
            namespace=namespace,
            max_entries=max_entries,
            ttl_seconds=ttl_seconds,
        )

    raise ValueError("Cache backend must be memory or sqlite")