from production_rag import embeddings


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
