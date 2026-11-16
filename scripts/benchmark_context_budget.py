import os
from pathlib import Path
import logging

from qdrant_client import QdrantClient

from production_rag.answer_cache import BoundedAnswerCache
from production_rag.answering import answer_question
from production_rag.evaluation import load_evaluation_cases
from production_rag.providers import create_answer_providers


class BudgetTokenCollector(logging.Handler):
    def __init__(self):
        super().__init__()
        self.measurements = []
        self.provider_errors = []

    def emit(self, record):
        if record.getMessage() == "context_budget_applied":
            self.measurements.append(
                (record.prompt_tokens_before, record.prompt_tokens_after)
            )
        if record.getMessage() in {"provider_call_failed", "provider_exhausted"}:
            self.provider_errors.append(
                (
                    record.getMessage(),
                    getattr(record, "provider", "unknown"),
                    getattr(record, "error_type", "unknown"),
                )
            )


token_logger = logging.getLogger("production_rag.answering")
token_logger.setLevel(logging.INFO)
token_collector = BudgetTokenCollector()
token_logger.addHandler(token_collector)
provider_logger = logging.getLogger("production_rag.providers")
provider_logger.setLevel(logging.INFO)
provider_logger.addHandler(token_collector)

client = QdrantClient(url=os.environ["QDRANT_URL"])
cases = load_evaluation_cases(Path("tests/fixtures/retrieval_eval.jsonl"))
answerable_cases = [case for case in cases if case.answerable]

primary_provider, backup_provider = create_answer_providers(
    os.environ["GOOGLE_API_KEY"],
    os.environ["GROQ_API_KEY"],
)


def run_condition(apply_context_budget: bool) -> tuple[int, int, int, int]:
    relevant_citation_answers = 0
    measurement_start = len(token_collector.measurements)
    for case in answerable_cases[:1]:
        result = answer_question(
            qdrant_client=client,
            primary_provider=primary_provider,
            backup_provider=backup_provider,
            question=case.question,
            tenant_id="benchmark-tenant",
            user_id="benchmark-user",
            ready_document_ids=None,
            answer_cache=BoundedAnswerCache(),
            document_set_version=f"budget-{apply_context_budget}",
            role="member",
            apply_context_budget=apply_context_budget,
            context_token_limit=500,
        )
        cited_ids = {citation.chunk_id for citation in result.citations}
        if cited_ids & case.relevant_chunk_ids:
            relevant_citation_answers += 1

    measurements = token_collector.measurements[measurement_start:]
    prompt_tokens_before = sum(before for before, _ in measurements)
    prompt_tokens_after = sum(after for _, after in measurements)

    return (
        relevant_citation_answers,
        min(1, len(answerable_cases)),
        prompt_tokens_before,
        prompt_tokens_after,
    )


for enabled in (True, False):
    hits, total, before, after = run_condition(enabled)
    mode = "budget_on" if enabled else "budget_off"
    print(
        f"{mode}: relevant_citations={hits}/{total}, "
        f"prompt_tokens_before={before}, prompt_tokens_after={after}"
    )
    print(f"provider_errors={token_collector.provider_errors}")

client.close()
