from production_rag import context_budget
from types import SimpleNamespace


def test_estimate_prompt_tokens_counts_formatted_messages(monkeypatch):
    monkeypatch.setattr(
        context_budget,
        "get_tokenizer",
        lambda: "fake-tokenizer",
    )

    def fake_count_tokens(text, tokenizer):
        assert text == "system: Be brief.\n\nuser: Hello"
        assert tokenizer == "fake-tokenizer"
        return 5

    monkeypatch.setattr(context_budget, "count_tokens", fake_count_tokens)

    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello"},
    ]

    assert context_budget.estimate_prompt_tokens(messages) == 5


def test_remove_near_duplicates_keeps_best_rank_and_other_documents():
    best_ranked = SimpleNamespace(
        payload={
            "document_id": "doc-a",
            "text": "Enable multifactor authentication for administrator accounts.",
        }
    )
    near_copy = SimpleNamespace(
        payload={
            "document_id": "doc-a",
            "text": "Enable multi-factor authentication for administrator accounts.",
        }
    )
    other_document = SimpleNamespace(
        payload={
            "document_id": "doc-b",
            "text": "Enable multifactor authentication for administrator accounts.",
        }
    )

    kept = context_budget.remove_near_duplicate_chunks(
        [best_ranked, near_copy, other_document]
    )

    assert kept == [best_ranked, other_document]


def test_select_chunks_stops_at_first_chunk_that_does_not_fit(monkeypatch):
    monkeypatch.setattr(
        context_budget,
        "get_tokenizer",
        lambda: "fake-tokenizer",
    )
    token_costs = {
        "[chunk-1]\nfirst": 3,
        "[chunk-2]\nsecond": 6,
        "[chunk-3]\nthird": 1,
    }
    monkeypatch.setattr(
        context_budget,
        "count_tokens",
        lambda text, tokenizer: token_costs[text],
    )

    first = SimpleNamespace(
        payload={
            "chunk_id": "chunk-1",
            "document_id": "doc-1",
            "location": "page 2",
            "text": "first",
        }
    )
    second = SimpleNamespace(
        payload={
            "chunk_id": "chunk-2",
            "document_id": "doc-1",
            "location": "page 3",
            "text": "second",
        }
    )
    third = SimpleNamespace(
        payload={
            "chunk_id": "chunk-3",
            "document_id": "doc-1",
            "location": "page 4",
            "text": "third",
        }
    )

    selected = context_budget.select_chunks_within_context_budget(
        [first, second, third],
        max_context_tokens=5,
    )

    assert selected == [first]
    assert selected[0].payload["location"] == "page 2"
