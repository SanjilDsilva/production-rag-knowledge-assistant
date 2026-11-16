from math import log2
from dataclasses import dataclass
import json
from pathlib import Path
from qdrant_client import QdrantClient

from production_rag.embeddings import embed_query
from production_rag.vector_store import search_chunks
from production_rag.answering import UNSUPPORTED_ANSWER


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    category: str
    relevant_chunk_ids: set[str]
    answerable: bool


def load_evaluation_cases(path: Path) -> list[EvaluationCase]:
    valid_categories = {
        "exact_word",
        "paraphrased",
        "multi_document",
        "unanswerable",
    }
    cases = []

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error

            if not isinstance(record, dict):
                raise ValueError(f"Line {line_number} must contain an object")

            question = record.get("question")
            category = record.get("category")
            relevant_ids = record.get("relevant_chunk_ids")
            answerable = record.get("answerable")

            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Line {line_number} has no question")
            if not isinstance(category, str) or category not in valid_categories:
                raise ValueError(f"Line {line_number} has an unknown category")
            if not isinstance(relevant_ids, list) or any(
                not isinstance(chunk_id, str) or not chunk_id
                for chunk_id in relevant_ids
            ):
                raise ValueError(f"Line {line_number} has invalid chunk IDs")
            if not isinstance(answerable, bool):
                raise ValueError(f"Line {line_number} needs an answerable flag")
            if answerable != bool(relevant_ids):
                raise ValueError(
                    f"Line {line_number} answerable flag disagrees with its chunk IDs"
                )
            if (category == "unanswerable") == answerable:
                raise ValueError(
                    f"Line {line_number} category disagrees with answerable flag"
                )

            cases.append(
                EvaluationCase(
                    question=question.strip(),
                    category=category,
                    relevant_chunk_ids=set(relevant_ids),
                    answerable=answerable,
                )
            )

    if not cases:
        raise ValueError("Evaluation file contains no cases")

    return cases


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    top_results = set(retrieved_ids[:k])
    found_relevant = top_results & relevant_ids

    return len(found_relevant) / len(relevant_ids)


def reciprocal_rank_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_by_question: list[list[str]],
    relevant_by_question: list[set[str]],
    k: int,
) -> float:
    if len(retrieved_by_question) != len(relevant_by_question):
        raise ValueError("Each question needs a relevant ID set")

    if not retrieved_by_question:
        raise ValueError("At least one question is required")

    scores = [
        reciprocal_rank_at_k(retrieved_ids, relevant_ids, k)
        for retrieved_ids, relevant_ids in zip(
            retrieved_by_question,
            relevant_by_question,
        )
    ]

    return sum(scores) / len(scores)


def ndcg_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    if k < 1:
        raise ValueError("k must be at least 1")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty")

    actual_score = 0.0
    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            actual_score += 1 / log2(rank + 1)

    ideal_count = min(len(relevant_ids), k)

    best_possible_score = sum(1 / log2(rank + 1) for rank in range(1, ideal_count + 1))

    return actual_score / best_possible_score


def evaluate_retrieval(
    retrieved_by_question: list[list[str]],
    relevant_by_question: list[set[str]],
    k: int,
) -> dict[str, float]:

    if len(retrieved_by_question) != len(relevant_by_question):
        raise ValueError("Each question needs a relevant ID set")

    if not retrieved_by_question:
        raise ValueError("At least one question is required")

    recall_scores = []
    ndcg_scores = []

    for retrieved_ids, relevant_ids in zip(
        retrieved_by_question,
        relevant_by_question,
    ):
        recall_scores.append(recall_at_k(retrieved_ids, relevant_ids, k))
        ndcg_scores.append(ndcg_at_k(retrieved_ids, relevant_ids, k))

    return {
        f"recall@{k}": sum(recall_scores) / len(recall_scores),
        f"mrr@{k}": mean_reciprocal_rank(
            retrieved_by_question,
            relevant_by_question,
            k,
        ),
        f"ndcg@{k}": sum(ndcg_scores) / len(ndcg_scores),
    }


def run_retrieval_evaluation(
    client: QdrantClient,
    cases: list[EvaluationCase],
    tenant_id: str,
    user_id: str,
    k: int = 5,
    search_mode: str = "hybrid",
    verbose: bool = False,
) -> dict[str, float]:
    if k < 1:
        raise ValueError("k must be at least 1")

    answerable_cases = [case for case in cases if case.answerable]

    if not answerable_cases:
        raise ValueError("At least one answerable case is required")

    retrieved_by_question = []
    relevant_by_question = []

    for case in answerable_cases:
        query_vector = embed_query(case.question)
        results = search_chunks(
            client=client,
            query_text=case.question,
            query_vector=query_vector,
            tenant_id=tenant_id,
            user_id=user_id,
            limit=k,
            search_mode=search_mode,
        )

        retrieved_ids = []
        for result in results:
            payload = result.payload or {}
            chunk_id = payload.get("chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Search result is missing its chunk ID")
            retrieved_ids.append(chunk_id)

        if verbose:
            print(
                f"{case.category}: question={case.question!r} "
                f"expected={sorted(case.relevant_chunk_ids)}"
                f"retrieved={retrieved_ids}"
            )
        retrieved_by_question.append(retrieved_ids)
        relevant_by_question.append(case.relevant_chunk_ids)

    return evaluate_retrieval(retrieved_by_question, relevant_by_question, k)


def evaluate_answerability(
    cases: list[EvaluationCase],
    answers: list[str],
) -> dict[str, float]:
    if len(cases) != len(answers):
        raise ValueError("Each evaluation case needs one answer")

    if not cases:
        raise ValueError("At least one evaluation case is required")

    answerable_count = sum(case.answerable for case in cases)
    unanswerable_count = len(cases) - answerable_count
    if not answerable_count or not unanswerable_count:
        raise ValueError("Cases must include answerable and unanswerable questions")

    for answer in answers:
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError("Answers must be non-empty strings")

    covered = sum(
        case.answerable and answer.strip() != UNSUPPORTED_ANSWER
        for case, answer in zip(cases, answers)
    )
    correctly_abstained = sum(
        not case.answerable and answer.strip() == UNSUPPORTED_ANSWER
        for case, answer in zip(cases, answers)
    )

    return {
        "answerable_coverage": covered / answerable_count,
        "unanswerable_abstention_rate": (correctly_abstained / unanswerable_count),
    }
