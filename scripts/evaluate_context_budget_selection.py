import os
from pathlib import Path
from statistics import mean

from qdrant_client import QdrantClient

from production_rag.answering import build_messages
from production_rag.context_budget import (
    estimate_prompt_tokens,
    select_chunks_within_context_budget,
)
from production_rag.embeddings import embed_query
from production_rag.evaluation import EvaluationCase, load_evaluation_cases
from production_rag.vector_store import search_chunks


EVALUATION_PATH = Path("tests/fixtures/retrieval_eval.jsonl")
TENANT_ID = "benchmark-tenant"
USER_ID = "benchmark-user"
RETRIEVAL_LIMIT = 5
CONTEXT_TOKEN_LIMIT = 500
RETRIEVED_CHUNKS_BY_QUESTION = {}


def evaluate_case(
    client: QdrantClient,
    case: EvaluationCase,
    context_token_limit: int | None = CONTEXT_TOKEN_LIMIT,
) -> dict[str, int | float | None]:
    if case.question not in RETRIEVED_CHUNKS_BY_QUESTION:
        query_vector = embed_query(case.question)
        RETRIEVED_CHUNKS_BY_QUESTION[case.question] = search_chunks(
            client=client,
            query_text=case.question,
            query_vector=query_vector,
            tenant_id=TENANT_ID,
            user_id=USER_ID,
            limit=RETRIEVAL_LIMIT,
            search_mode="hybrid",
        )

    retrieved = RETRIEVED_CHUNKS_BY_QUESTION[case.question]
    if context_token_limit is None:
        selected = retrieved
    else:
        selected = select_chunks_within_context_budget(
            retrieved,
            context_token_limit,
        )

    retrieved_ids = {
        chunk.payload["chunk_id"]
        for chunk in retrieved
        if chunk.payload and "chunk_id" in chunk.payload
    }
    selected_ids = {
        chunk.payload["chunk_id"]
        for chunk in selected
        if chunk.payload and "chunk_id" in chunk.payload
    }
    relevant_ids = case.relevant_chunk_ids
    before_recall = (
        len(retrieved_ids & relevant_ids) / len(relevant_ids) if relevant_ids else None
    )
    after_recall = (
        len(selected_ids & relevant_ids) / len(relevant_ids) if relevant_ids else None
    )
    prompt_tokens_before = (
        estimate_prompt_tokens(build_messages(case.question, retrieved))
        if retrieved
        else 0
    )
    prompt_tokens_after = (
        estimate_prompt_tokens(build_messages(case.question, selected))
        if selected
        else 0
    )

    return {
        "prompt_tokens_before": prompt_tokens_before,
        "prompt_tokens_after": prompt_tokens_after,
        "evidence_recall_before": before_recall,
        "evidence_recall_after": after_recall,
    }


def main() -> None:
    cases = load_evaluation_cases(EVALUATION_PATH)
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    budgets = (None, 300, 500, 750, 3000)

    try:
        for budget in budgets:
            results = [evaluate_case(client, case, budget) for case in cases]
            answerable = [
                result
                for result in results
                if result["evidence_recall_before"] is not None
            ]
            lost_cases = [
                (case, result)
                for case, result in zip(cases, results)
                if result["evidence_recall_before"] is not None
                and result["evidence_recall_after"] < result["evidence_recall_before"]
            ]
            budget_name = "none" if budget is None else str(budget)
            print(
                f"\nbudget_tokens={budget_name} "
                f"questions={len(results)} answerable={len(answerable)}"
            )
            print(
                "avg_prompt_tokens_before="
                f"{mean(r['prompt_tokens_before'] for r in results):.1f}"
            )
            print(
                "avg_prompt_tokens_after="
                f"{mean(r['prompt_tokens_after'] for r in results):.1f}"
            )
            print(
                "avg_gold_recall_before="
                f"{mean(r['evidence_recall_before'] for r in answerable):.3f}"
            )
            print(
                "avg_gold_recall_after="
                f"{mean(r['evidence_recall_after'] for r in answerable):.3f}"
            )
            print(f"evidence_lost_cases={len(lost_cases)}")

            if budget == 500:
                for case, result in lost_cases:
                    print(f"  {case.category}: {case.question}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
