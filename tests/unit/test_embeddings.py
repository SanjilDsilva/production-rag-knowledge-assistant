from production_rag import embeddings
from collections import OrderedDict


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
            "Represent this sentence for searching relevant passages: What is the budget?"
        ],
        ["The budget is 500."],
    ]
    assert fake_model.normalize_flags == [True, True]
    assert query_vector == [1.0, 0.0]
    assert document_vectors == [[1.0, 0.0]]


def test_query_embedding_cache_reuses_and_evicts_oldest(monkeypatch):
    fake_model = FakeModel()
    monkeypatch.setattr(embeddings, "get_embedding_model", lambda: fake_model)
    monkeypatch.setattr(embeddings, "_query_cache", OrderedDict())
    monkeypatch.setattr(embeddings, "QUERY_CACHE_MAX_ENTRIES", 2)
    monkeypatch.setattr(embeddings, "_query_cache_hits", 0)
    monkeypatch.setattr(embeddings, "_query_cache_misses", 0)

    first_a = embeddings.embed_query("question a")
    first_a[0] = 9.0
    embeddings.embed_query("question b")
    repeated_a = embeddings.embed_query("question a")
    embeddings.embed_query("question c")
    reloaded_b = embeddings.embed_query("question b")

    assert repeated_a == [1.0, 0.0]
    assert reloaded_b == [1.0, 0.0]
    assert len(fake_model.received_texts) == 4
    assert len(embeddings._query_cache) == 2
    assert embeddings._query_cache_hits == 1
    assert embeddings._query_cache_misses == 4
