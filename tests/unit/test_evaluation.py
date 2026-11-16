from production_rag.evaluation import (
    recall_at_k,
    reciprocal_rank_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    evaluate_retrieval,
    load_evaluation_cases,
)
from production_rag.answering import UNSUPPORTED_ANSWER
import os
import math
import statistics
import time

from qdrant_client import QdrantClient, models
import pytest

from pathlib import Path

from types import SimpleNamespace
from production_rag import evaluation
from production_rag.evaluation import EvaluationCase, evaluate_answerability


def test_recall_at_k_counts_relevant_chunks_in_top_results():
    score = recall_at_k(
        ["chunk-x", "chunk-a", "chunk-b"],
        {"chunk-a", "chunk-b"},
        2,
    )

    assert score == 0.5


def test_reciprocal_rank_at_k_scores_position_of_first_relevant_chunk():
    score = reciprocal_rank_at_k(
        ["chunk-x", "chunk-a", "chunk-b"],
        {"chunk-a", "chunk-b"},
        3,
    )

    assert score == 0.5


def test_reciprocal_rank_at_k_is_zero_when_no_relevant_chunk_is_found():
    score = reciprocal_rank_at_k(
        ["chunk-x", "chunk-y"],
        {"chunk-a"},
        2,
    )

    assert score == 0.0


def test_mean_reciprocal_rank_averages_question_scores():
    score = mean_reciprocal_rank(
        [["chunk-a", "chunk-x"], ["chunk-y", "chunk-b"]],
        [{"chunk-a"}, {"chunk-b"}],
        2,
    )

    assert score == 0.75


def test_ndcg_at_k_rewards_relevant_chunks_near_the_top():
    score = ndcg_at_k(
        ["chunk-x", "chunk-a", "chunk-b"],
        {"chunk-a", "chunk-b"},
        3,
    )

    assert score == 0.6934264036172708


def test_evaluate_retrieval_returns_average_metrics():
    metrics = evaluate_retrieval(
        [["chunk-a", "chunk-x"], ["chunk-y", "chunk-b"]],
        [{"chunk-a"}, {"chunk-b"}],
        2,
    )

    assert metrics["recall@2"] == 1.0
    assert metrics["mrr@2"] == 0.75
    assert metrics["ndcg@2"] == pytest.approx(0.8154648767857288)


def test_load_evaluation_cases_reads_labeled_questions():
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_eval.jsonl"
    )

    cases = load_evaluation_cases(fixture_path)

    assert len(cases) >= 50
    assert cases[0].category == "exact_word"
    assert cases[0].answerable is True
    assert cases[0].relevant_chunk_ids == {
        "4ffb0e579b638a1d44e3747c1709b1118d0d811e11bb2b8750ee15477e347590"
    }
    assert cases[2].category == "multi_document"
    assert len(cases[2].relevant_chunk_ids) == 3
    assert cases[3].category == "unanswerable"
    assert cases[3].answerable is False
    assert cases[3].relevant_chunk_ids == set()


def test_retrieval_runner_scores_search_results(monkeypatch, capsys):
    cases = [
        EvaluationCase(
            question="exact question",
            category="exact_word",
            relevant_chunk_ids={"chunk-a"},
            answerable=True,
        ),
        EvaluationCase(
            question="paraphrased question",
            category="paraphrased",
            relevant_chunk_ids={"chunk-b"},
            answerable=True,
        ),
    ]
    search_calls = []

    monkeypatch.setattr(
        evaluation,
        "embed_query",
        lambda question: [0.0] * 384,
    )

    def fake_search(**kwargs):
        search_calls.append(kwargs)
        chunk_id = "chunk-a" if kwargs["query_text"] == "exact question" else "chunk-b"
        return [SimpleNamespace(payload={"chunk_id": chunk_id})]

    monkeypatch.setattr(evaluation, "search_chunks", fake_search)

    metrics = evaluation.run_retrieval_evaluation(
        client=object(),
        cases=cases,
        tenant_id="tenant-a",
        user_id="user-a",
        k=3,
        search_mode="dense",
        verbose=True,
    )

    assert metrics == {
        "recall@3": 1.0,
        "mrr@3": 1.0,
        "ndcg@3": 1.0,
    }
    assert len(search_calls) == 2
    assert all(call["tenant_id"] == "tenant-a" for call in search_calls)
    assert all(call["user_id"] == "user-a" for call in search_calls)
    output = capsys.readouterr().out
    assert "question='exact question'" in output
    assert "question='paraphrased question'" in output


LOCAL_QDRANT_DIR = Path(__file__).resolve().parents[2] / "qdrant_storage"


@pytest.mark.skipif(
    os.environ.get("RUN_BENCHMARK_INTEGRATION") != "1" or not LOCAL_QDRANT_DIR.exists(),
    reason="Opt-in check for the local benchmark Qdrant database",
)
def test_local_benchmark_retrieval_reports_metrics(monkeypatch):
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "retrieval_eval.jsonl"
    )
    cases = load_evaluation_cases(fixture_path)
    client = QdrantClient(path=str(LOCAL_QDRANT_DIR))

    try:
        benchmark_count = client.count(
            collection_name="knowledge_chunks",
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value="benchmark-tenant"),
                    ),
                    models.FieldCondition(
                        key="user_id",
                        match=models.MatchValue(value="benchmark-user"),
                    ),
                    models.FieldCondition(
                        key="source",
                        match=models.MatchValue(value="benchmark"),
                    ),
                ]
            ),
            exact=True,
        )
        assert benchmark_count.count == 10_792

        search_latencies_ms = {mode: [] for mode in ("dense", "keyword", "hybrid")}
        warmed_modes = set()
        original_search = evaluation.search_chunks

        def timed_search(**kwargs):
            mode = kwargs["search_mode"]
            started_at = time.perf_counter()
            results = original_search(**kwargs)
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            if mode in warmed_modes:
                search_latencies_ms[mode].append(elapsed_ms)
            else:
                warmed_modes.add(mode)

            return results

        monkeypatch.setattr(evaluation, "search_chunks", timed_search)

        for search_mode in ("dense", "keyword", "hybrid"):
            metrics = evaluation.run_retrieval_evaluation(
                client=client,
                cases=cases,
                tenant_id="benchmark-tenant",
                user_id="benchmark-user",
                k=5,
                search_mode=search_mode,
            )
            print(f"{search_mode}: {metrics}")

            samples = sorted(search_latencies_ms[search_mode])
            p50_ms = statistics.median(samples)
            p95_index = math.ceil(0.95 * len(samples)) - 1
            p95_ms = samples[p95_index]
            print(
                f"{search_mode} search-only latency: "
                f"p50={p50_ms:.2f} ms, p95={p95_ms:.2f} ms, "
                f"n={len(samples)}"
            )

            assert set(metrics) == {"recall@5", "mrr@5", "ndcg@5"}
            assert metrics["recall@5"] > 0
    finally:
        client.close()


def test_answerability_metrics_measure_coverage_and_abstention():
    cases = [
        EvaluationCase(
            question="Where is the vulnerable endpoint?",
            category="exact_word",
            relevant_chunk_ids={"chunk-1"},
            answerable=True,
        ),
        EvaluationCase(
            question="What are the support desk's opening hours?",
            category="unanswerable",
            relevant_chunk_ids=set(),
            answerable=False,
        ),
    ]
    answers = [
        "The vulnerable endpoint is /api/example.",
        UNSUPPORTED_ANSWER,
    ]

    metrics = evaluate_answerability(cases, answers)

    assert metrics == {
        "answerable_coverage": 1.0,
        "unanswerable_abstention_rate": 1.0,
    }
