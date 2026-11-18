import json
from pathlib import Path
from types import SimpleNamespace

from scripts.benchmark_provider_settings import (
    MeasurementCollector,
    chunk_ids_for_diagnostics,
    measure_gold_chunk_coverage,
    output_paths,
    print_condition_summary,
)


def test_answer_diagnostics_writer_saves_flags_and_rate_limit_info(tmp_path):
    collector = MeasurementCollector()
    collector.start_question(condition_id=3, question_id=7)
    collector.provider_events.append(
        SimpleNamespace(
            rate_limit_failure=True,
            status_code=429,
            retry_after_seconds=5.0,
            rate_limit_headers={"x-ratelimit-limit-requests": "100"},
        )
    )
    measurement = {
        "answerable": True,
        "fallback_used": True,
        "degraded_mode": False,
        "supplied_chunk_ids": ["source-b", "source-a"],
        "retry_wait_ms": 125.0,
        "provider_call_succeeded": True,
        "finish_reason": "stop",
        "answer_empty": False,
        "answer_contains_citation": True,
        "citation_valid": True,
        "citation_in_acceptable_set": True,
        "exact_gold_match": True,
        "error_classes": ["RateLimitError"],
        "raw_answers": ["Approved [1]."],
        "cited_chunk_ids": ["gold-a"],
        "gold_cited_chunk_ids": ["gold-a"],
        "gold_chunk_coverage": 0.5,
        "all_gold_chunks_cited": False,
        "acceptable_cited_chunk_ids": ["gold-a"],
        "unaccepted_cited_chunk_ids": [],
        "all_citations_acceptable": True,
    }
    path = tmp_path / "diagnostics.jsonl"

    collector.write_answer_diagnostics(path, measurement)
    row = json.loads(path.read_text(encoding="utf-8"))

    assert row["question_id"] == 7
    assert row["fallback_used"] is True
    assert row["degraded_mode"] is False
    assert row["retry_wait_ms"] == 125.0
    assert row["supplied_chunk_ids"] == ["source-b", "source-a"]
    assert row["citation_valid"] is True
    assert row["raw_answers"] == ["Approved [1]."]
    assert row["rate_limit_events"][0]["status_code"] == 429
    assert row["rate_limit_events"][0]["retry_after_seconds"] == 5.0
    assert row["cited_chunk_ids"] == ["gold-a"]
    assert row["gold_cited_chunk_ids"] == ["gold-a"]
    assert row["gold_chunk_coverage"] == 0.5
    assert row["all_gold_chunks_cited"] is False
    assert row["acceptable_cited_chunk_ids"] == ["gold-a"]
    assert row["unaccepted_cited_chunk_ids"] == []
    assert row["all_citations_acceptable"] is True


def test_collector_tracks_retry_wait_separately():
    collector = MeasurementCollector()
    collector.start_question(condition_id=3, question_id=7)
    collector.emit(
        SimpleNamespace(
            getMessage=lambda: "provider_retry_wait_completed",
            provider="GroqAnswerProvider",
            elapsed_ms=125.0,
            stage="model_generation",
        )
    )

    assert collector.retry_wait_ms == 125.0
    assert collector.retry_wait_by_stage == {"model_generation": 125.0}


def test_summary_reports_retry_wait_and_degraded_count(capsys):
    measurement = {
        "total_ms": 900.0,
        "provider_attempt_ms": [200.0],
        "failures": 1,
        "rate_limits": 1,
        "answerable": True,
        "citation_in_acceptable_set": True,
        "all_gold_chunks_cited": True,
        "all_citations_acceptable": True,
        "gold_chunk_coverage": 1.0,
        "correct_abstention": 0,
        "fallback_used": False,
        "degraded_mode": True,
        "retry_wait_ms": 700.0,
        "stages": {
            "query_embedding": 10.0,
            "retrieval": 20.0,
            "model_generation": 190.0,
            "citation_validation": 1.0,
        },
    }

    print_condition_summary("groq_default", [measurement])
    summary = capsys.readouterr().out

    assert "degraded_mode=1" in summary
    assert "retry_wait_ms: p50=700.0, p95=700.0" in summary
    assert "model_generation_active_ms: p50=190.0" in summary


def test_chunk_ids_for_diagnostics_preserves_prompt_order():
    chunks = [
        SimpleNamespace(payload={"chunk_id": "source-b", "text": "private text"}),
        SimpleNamespace(payload={"chunk_id": "source-a", "text": "more text"}),
    ]

    assert chunk_ids_for_diagnostics(chunks) == ["source-b", "source-a"]


def test_output_paths_use_separate_safe_run_tags(monkeypatch):
    monkeypatch.setenv("PROVIDER_BENCHMARK_TAG", "run_2")

    assert output_paths() == (
        Path("data/provider_call_timings_run_2.jsonl"),
        Path("data/provider_answer_diagnostics_run_2.jsonl"),
    )


def test_gold_chunk_coverage_reports_partial_and_complete_citations():
    relevant = {"gold-a", "gold-b", "gold-c"}

    cited_ids, coverage, complete = measure_gold_chunk_coverage(
        {"gold-a", "gold-b"}, relevant
    )
    assert cited_ids == ["gold-a", "gold-b"]
    assert coverage == 0.667
    assert complete is False

    cited_ids, coverage, complete = measure_gold_chunk_coverage(relevant, relevant)
    assert cited_ids == ["gold-a", "gold-b", "gold-c"]
    assert coverage == 1.0
    assert complete is True
