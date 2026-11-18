from pathlib import Path
import csv

from production_rag.evaluation import (
    EvaluationCase,
    load_evaluation_cases,
)

import logging
import time
import json
from production_rag.answer_cache import BoundedAnswerCache
import production_rag.answering as answering
from qdrant_client import QdrantClient
import production_rag.embeddings as embeddings
import re

EVALUATION_PATH = Path("tests/fixtures/retrieval_eval.jsonl")
REVIEW_PATH = Path("data/provider_support_review.csv")
PILOT_CATEGORY_COUNTS = {
    "exact_word": 4,
    "multi_document": 3,
    "paraphrased": 3,
}


def select_pilot_cases(
    cases: list[EvaluationCase],
) -> list[EvaluationCase]:
    selected = []
    for category, count in PILOT_CATEGORY_COUNTS.items():
        matching = [
            case for case in cases if case.answerable and case.category == category
        ]
        if len(matching) < count:
            raise ValueError(f"Not enough answerable {category} cases")
        selected.extend(matching[:count])

    return selected


def load_accepted_chunk_ids_by_question(path: Path) -> dict[int, set[str]]:
    accepted_by_question = {}

    with path.open(encoding="utf-8", newline="") as review_file:
        for row in csv.DictReader(review_file):
            if row.get("review_decision", "").strip().casefold() != "accept":
                continue

            question_id = int(row["question_id"])
            chunk_id = row["chunk_id"]
            accepted_by_question.setdefault(question_id, set()).add(chunk_id)

    if not accepted_by_question:
        raise ValueError("Support review file has no accepted chunks")

    return accepted_by_question


def measure_gold_chunk_coverage(
    cited_chunk_ids: set[str],
    relevant_chunk_ids: set[str],
) -> tuple[list[str], float, bool]:
    cited_gold_ids = sorted(cited_chunk_ids & relevant_chunk_ids)
    if not relevant_chunk_ids:
        return cited_gold_ids, 0.0, False

    coverage = len(cited_gold_ids) / len(relevant_chunk_ids)
    all_gold_cited = relevant_chunk_ids.issubset(cited_gold_ids)
    return cited_gold_ids, round(coverage, 3), all_gold_cited


def chunk_ids_for_diagnostics(chunks) -> list[str]:
    chunk_ids = []
    for chunk in chunks:
        chunk_id = (chunk.payload or {}).get("chunk_id")
        if isinstance(chunk_id, str):
            chunk_ids.append(chunk_id)
    return chunk_ids


import math


CONDITIONS = ((3, "groq_default"),)
QUESTION_PAUSE_SECONDS = 15


def output_paths() -> tuple[Path, Path]:
    tag = os.environ.get("PROVIDER_BENCHMARK_TAG", "default")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
        raise ValueError("PROVIDER_BENCHMARK_TAG may contain only letters, numbers, _ and -")
    return (
        Path(f"data/provider_call_timings_{tag}.jsonl"),
        Path(f"data/provider_answer_diagnostics_{tag}.jsonl"),
    )


def percentile(values: list[float], percent: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(percent / 100 * len(ordered)) - 1
    return round(ordered[max(index, 0)], 2)


class DisabledFallbackProvider:
    model_name = "disabled"

    def generate_answer(self, prompt):
        raise RuntimeError("Fallback disabled for isolated measurement")


def configure_condition(gemini, groq, condition_name):
    is_low = condition_name.endswith("_low")
    is_gemini = condition_name.startswith("gemini")

    gemini.thinking_level = "low" if is_gemini and is_low else None
    groq.reasoning_effort = "low" if not is_gemini and is_low else None

    return gemini if is_gemini else groq


def run_one_question(client, provider, fallback, case, collector, acceptable_chunk_ids):
    embeddings._query_cache.clear()
    started = time.perf_counter()
    raw_answers = []
    supplied_chunk_ids = []
    original_generate = answering.generate_with_fallback
    original_selector = answering.select_chunks_within_context_budget

    def capture_answer(**kwargs):
        provider_result = original_generate(**kwargs)
        raw_answers.append(provider_result.text)
        return provider_result

    def capture_selected_chunks(chunks, max_context_tokens):
        selected = original_selector(chunks, max_context_tokens)
        supplied_chunk_ids[:] = chunk_ids_for_diagnostics(selected)
        return selected

    answering.generate_with_fallback = capture_answer
    answering.select_chunks_within_context_budget = capture_selected_chunks
    try:
        result = answering.answer_question(
            qdrant_client=client,
            primary_provider=provider,
            backup_provider=fallback,
            question=case.question,
            tenant_id="benchmark-tenant",
            user_id="benchmark-user",
            ready_document_ids=None,
            answer_cache=BoundedAnswerCache(),
            document_set_version="provider-settings-pilot",
            role="member",
            retry_wait_budget=RetryWaitBudget(max_wait_seconds=None),
        )
    finally:
        answering.generate_with_fallback = original_generate
        answering.select_chunks_within_context_budget = original_selector
    total_ms = (time.perf_counter() - started) * 1000
    cited_ids = {citation.chunk_id for citation in result.citations}
    real_events = [
        event
        for event in collector.provider_events
        if getattr(event, "provider", "")
        in {"GeminiAnswerProvider", "GroqAnswerProvider"}
        and event.getMessage()
        in {"provider_call_succeeded", "provider_call_failed"}
    ]
    successful_events = [
        event
        for event in real_events
        if event.getMessage() == "provider_call_succeeded"
    ]
    failed_events = [
        event for event in real_events if event.getMessage() == "provider_call_failed"
    ]
    raw_answer = raw_answers[-1] if raw_answers else ""
    contains_citation = bool(
        re.search(
            r"\[\d+(?:,\s*\d+)*\]|【\d+(?:,\s*\d+)*】", raw_answer
        )
    )
    citation_valid = bool(
        raw_answer and result.answer == raw_answer and result.citations
    )
    observed_cited_ids = cited_ids if citation_valid else set()
    gold_cited_ids, gold_coverage, all_gold_cited = measure_gold_chunk_coverage(
        observed_cited_ids, case.relevant_chunk_ids
    )
    acceptable_cited_ids = sorted(observed_cited_ids & acceptable_chunk_ids)
    unaccepted_cited_ids = sorted(observed_cited_ids - acceptable_chunk_ids)
    exact_gold_match = bool(gold_cited_ids)
    acceptable_match = bool(acceptable_cited_ids)
    all_citations_acceptable = citation_valid and not unaccepted_cited_ids
    adjusted_stages = dict(collector.stage_ms)
    for stage, waited_ms in collector.retry_wait_by_stage.items():
        if stage in adjusted_stages:
            adjusted_stages[stage] = max(0.0, adjusted_stages[stage] - waited_ms)
    return {
        "answerable": case.answerable,
        "fallback_used": result.fallback_used,
        "degraded_mode": result.degraded_mode,
        "supplied_chunk_ids": supplied_chunk_ids,
        "total_ms": round(total_ms, 2),
        "stages": adjusted_stages,
        "retry_wait_ms": round(collector.retry_wait_ms, 2),
        "correct_abstention": int(
            not case.answerable and result.answer == answering.UNSUPPORTED_ANSWER
        ),
        "failures": sum(
            event.getMessage() == "provider_call_failed" for event in real_events
        ),
        "rate_limits": sum(
            bool(getattr(event, "rate_limit_failure", False)) for event in real_events
        ),
        "provider_attempt_ms": [
            round(float(event.elapsed_ms), 2) for event in real_events
        ],
        "raw_answers": raw_answers,
        "provider_call_succeeded": bool(successful_events),
        "finish_reason": (
            getattr(successful_events[-1], "finish_reason", None)
            if successful_events
            else None
        ),
        "answer_empty": bool(raw_answers) and not raw_answer.strip(),
        "answer_contains_citation": contains_citation,
        "citation_valid": citation_valid,
        "cited_chunk_ids": sorted(observed_cited_ids),
        "gold_cited_chunk_ids": gold_cited_ids,
        "gold_chunk_coverage": gold_coverage,
        "all_gold_chunks_cited": all_gold_cited,
        "acceptable_cited_chunk_ids": acceptable_cited_ids,
        "unaccepted_cited_chunk_ids": unaccepted_cited_ids,
        "all_citations_acceptable": all_citations_acceptable,
        "citation_in_acceptable_set": bool(acceptable_match),
        "exact_gold_match": bool(exact_gold_match),
        "error_classes": [
            getattr(event, "error_type", "unknown") for event in failed_events
        ],
    }


class MeasurementCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.run_id = time.time_ns()
        self.condition_id = 0
        self.question_id = 0
        self.stage_ms = {}
        self.retry_wait_ms = 0.0
        self.retry_wait_by_stage = {}
        self.provider_events = []
        self.call_id = 0

    def start_question(self, condition_id, question_id):
        self.condition_id = condition_id
        self.question_id = question_id
        self.stage_ms = {}
        self.retry_wait_ms = 0.0
        self.retry_wait_by_stage = {}
        self.provider_events = []

    def emit(self, record):
        event = record.getMessage()
        if event == "answer_stage_timing":
            self.stage_ms[record.stage] = record.elapsed_ms
        elif event == "provider_retry_wait_completed":
            waited_ms = float(getattr(record, "elapsed_ms", 0.0))
            stage = getattr(record, "stage", "model_generation")
            self.retry_wait_ms += waited_ms
            self.retry_wait_by_stage[stage] = (
                self.retry_wait_by_stage.get(stage, 0.0) + waited_ms
            )
            self.provider_events.append(record)
        elif event in {
            "provider_call_succeeded",
            "provider_call_failed",
        }:
            self.provider_events.append(record)

    def write_provider_events(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as output_file:
            for record in self.provider_events:
                provider = getattr(record, "provider", "unknown")
                if provider not in {
                    "GeminiAnswerProvider",
                    "GroqAnswerProvider",
                }:
                    continue

                event = record.getMessage()
                if event == "provider_retry_wait_completed":
                    row = {
                        "run_id": self.run_id,
                        "condition_id": self.condition_id,
                        "question_id": self.question_id,
                        "provider_id": provider,
                        "event_type": "retry_wait",
                        "stage": getattr(record, "stage", None),
                        "attempt": int(getattr(record, "attempt", 0)),
                        "elapsed_ms": float(getattr(record, "elapsed_ms", 0)),
                        "retry_after_seconds": getattr(
                            record, "retry_after_seconds", None
                        ),
                    }
                    output_file.write(json.dumps(row) + "\n")
                    continue
                if event not in {
                    "provider_call_succeeded",
                    "provider_call_failed",
                }:
                    continue

                self.call_id += 1
                status = getattr(record, "status_code", None)
                row = {
                    "run_id": self.run_id,
                    "call_id": self.call_id,
                    "condition_id": self.condition_id,
                    "question_id": self.question_id,
                    "event_type": "provider_call",
                    "provider_id": provider,
                    "model_id": getattr(record, "model", "unknown"),
                    "attempt": int(getattr(record, "attempt", 0)),
                    "elapsed_ms": float(getattr(record, "elapsed_ms", 0)),
                    "status_code": status if isinstance(status, int) else -1,
                    "input_tokens": getattr(record, "input_tokens", None),
                    "output_tokens": getattr(record, "output_tokens", None),
                    "thinking_tokens": getattr(record, "thinking_tokens", None),
                    "rate_limit_count": int(
                        bool(getattr(record, "rate_limit_failure", False))
                    ),
                    "finish_reason": getattr(record, "finish_reason", None),
                    "error_type": getattr(record, "error_type", None),
                    "error_message": getattr(record, "error_message", None),
                    "retry_after_seconds": getattr(record, "retry_after_seconds", None),
                    "rate_limit_headers": getattr(record, "rate_limit_headers", {}),
                }
                output_file.write(json.dumps(row) + "\n")

    def write_answer_diagnostics(self, path: Path, measurement: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        rate_limit_events = [
            {
                "status_code": getattr(event, "status_code", None),
                "retry_after_seconds": getattr(event, "retry_after_seconds", None),
                "headers": getattr(event, "rate_limit_headers", {}),
            }
            for event in self.provider_events
            if getattr(event, "rate_limit_failure", False)
        ]
        row = {
            "run_id": self.run_id,
            "condition_id": self.condition_id,
            "question_id": self.question_id,
            "answerable": measurement["answerable"],
            "fallback_used": measurement["fallback_used"],
            "degraded_mode": measurement["degraded_mode"],
            "supplied_chunk_ids": measurement["supplied_chunk_ids"],
            "retry_wait_ms": measurement["retry_wait_ms"],
            "provider_call_succeeded": measurement["provider_call_succeeded"],
            "finish_reason": measurement["finish_reason"],
            "answer_empty": measurement["answer_empty"],
            "answer_contains_citation": measurement["answer_contains_citation"],
            "citation_valid": measurement["citation_valid"],
            "citation_in_acceptable_set": measurement["citation_in_acceptable_set"],
            "exact_gold_match": measurement["exact_gold_match"],
            "cited_chunk_ids": measurement["cited_chunk_ids"],
            "gold_cited_chunk_ids": measurement["gold_cited_chunk_ids"],
            "gold_chunk_coverage": measurement["gold_chunk_coverage"],
            "all_gold_chunks_cited": measurement["all_gold_chunks_cited"],
            "acceptable_cited_chunk_ids": measurement["acceptable_cited_chunk_ids"],
            "unaccepted_cited_chunk_ids": measurement["unaccepted_cited_chunk_ids"],
            "all_citations_acceptable": measurement["all_citations_acceptable"],
            "error_classes": measurement["error_classes"],
            "rate_limit_events": rate_limit_events,
            "raw_answers": measurement["raw_answers"],
        }
        with path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(row) + "\n")


import os
from production_rag.providers import RetryWaitBudget, create_answer_providers


def print_condition_summary(condition_name, measurements):
    request_times = [item["total_ms"] for item in measurements]
    provider_times = [
        elapsed for item in measurements for elapsed in item["provider_attempt_ms"]
    ]
    failures = sum(item["failures"] for item in measurements)
    rate_limits = sum(item["rate_limits"] for item in measurements)

    answerable = [item for item in measurements if item["answerable"]]
    any_acceptable = sum(item["citation_in_acceptable_set"] for item in answerable)
    all_gold = sum(item["all_gold_chunks_cited"] for item in answerable)
    all_acceptable = sum(item["all_citations_acceptable"] for item in answerable)
    mean_gold_coverage = (
        sum(item["gold_chunk_coverage"] for item in answerable) / len(answerable)
        if answerable
        else 0.0
    )
    correct_abstentions = sum(item["correct_abstention"] for item in measurements)
    fallback_used_count = sum(item["fallback_used"] for item in measurements)
    degraded_mode_count = sum(item["degraded_mode"] for item in measurements)

    print(
        f"{condition_name}: failures={failures}, rate_limits={rate_limits}, "
        f"any_acceptable_citation={any_acceptable}/{len(answerable)}, "
        f"all_gold_chunks_cited={all_gold}/{len(answerable)}, "
        f"all_citations_acceptable={all_acceptable}/{len(answerable)}, "
        f"mean_gold_chunk_coverage={mean_gold_coverage:.3f}, "
        f"correct_abstentions={correct_abstentions}, "
        f"fallback_used={fallback_used_count}, "
        f"degraded_mode={degraded_mode_count}"
    )
    print(
        f"  end_to_end_ms (includes waits): p50={percentile(request_times, 50)}, "
        f"p95={percentile(request_times, 95)}"
    )
    print(
        f"  provider_call_ms: calls={len(provider_times)}, "
        f"p50={percentile(provider_times, 50)}, "
        f"p95={percentile(provider_times, 95)}"
    )
    wait_times = [item["retry_wait_ms"] for item in measurements]
    print(
        f"  retry_wait_ms: p50={percentile(wait_times, 50)}, "
        f"p95={percentile(wait_times, 95)}"
    )

    for stage in (
        "query_embedding",
        "retrieval",
        "model_generation",
        "citation_validation",
    ):
        values = [
            item["stages"][stage] for item in measurements if stage in item["stages"]
        ]
        stage_label = (
            "model_generation_active" if stage == "model_generation" else stage
        )
        print(
            f"  {stage_label}_ms: p50={percentile(values, 50)}, "
            f"p95={percentile(values, 95)}"
        )


def main():
    timing_path, diagnostics_path = output_paths()
    cases = load_evaluation_cases(EVALUATION_PATH)
    pilot_cases = select_pilot_cases(cases)
    reviewed_pilot_ids = load_accepted_chunk_ids_by_question(REVIEW_PATH)
    accepted_by_question = {
        case.question: reviewed_pilot_ids[question_id]
        for question_id, case in enumerate(pilot_cases, start=1)
    }
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    gemini, groq = create_answer_providers(
        os.environ["GOOGLE_API_KEY"],
        os.environ["GROQ_API_KEY"],
    )
    fallback = DisabledFallbackProvider()
    collector = MeasurementCollector()
    loggers = [
        logging.getLogger("production_rag.answering"),
        logging.getLogger("production_rag.providers"),
    ]
    for logger in loggers:
        logger.addHandler(collector)
        logger.setLevel(logging.INFO)

    try:
        for condition_id, condition_name in CONDITIONS:
            provider = configure_condition(gemini, groq, condition_name)
            measurements = []

            for question_id, case in enumerate(cases, start=1):
                collector.start_question(condition_id, question_id)
                measurement = run_one_question(
                    client,
                    provider,
                    fallback,
                    case,
                    collector,
                    accepted_by_question.get(case.question, case.relevant_chunk_ids),
                )
                measurements.append(measurement)
                collector.write_provider_events(timing_path)
                collector.write_answer_diagnostics(
                    diagnostics_path,
                    measurement,
                )
                print(
                    f"{condition_name}: question {question_id}/{len(cases)} complete"
                )
                if question_id < len(cases):
                    time.sleep(QUESTION_PAUSE_SECONDS)

            print_condition_summary(condition_name, measurements)
    finally:
        for logger in loggers:
            logger.removeHandler(collector)
        client.close()


if __name__ == "__main__":
    main()
