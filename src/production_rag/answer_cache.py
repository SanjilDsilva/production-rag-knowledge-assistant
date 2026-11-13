import hashlib
import json
from collections import OrderedDict
from threading import Lock
from time import monotonic


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().split())


def make_answer_cache_key(
    tenant_id: str,
    user_id: str,
    question: str,
    document_set_version: str,
    prompt_version: str,
    model_name: str,
) -> str:
    key_parts = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "question": normalize_question(question),
        "document_set_version": document_set_version,
        "prompt_version": prompt_version,
        "model_name": model_name,
    }

    encoded_key = json.dumps(
        key_parts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded_key).hexdigest()


class BoundedAnswerCache:
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
        self._entries: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self._lock = Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> str | None:
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
            total= self._hits + self._misses
            hit_rate = self._hits / total if total else 0.0

            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
            }


    def set(self, key: str, answer: str) -> None:
        with self._lock:
            expires_at = monotonic() + self.ttl_seconds
            self._entries[key] = (answer, expires_at)
            self._entries.move_to_end(key)

            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
