import production_rag.answer_cache as answer_cache


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
