import os
from pathlib import Path
import re
import csv

from qdrant_client import QdrantClient

from production_rag.embeddings import embed_query
from production_rag.evaluation import load_evaluation_cases
from production_rag.vector_store import search_chunks
from scripts.benchmark_provider_settings import select_pilot_cases

EVALUATION_PATH = Path("tests/fixtures/retrieval_eval.jsonl")
OUTPUT_PATH = Path("data/provider_support_review.csv")
SEARCH_MODES = ("dense", "keyword", "hybrid")


def collect_review_rows():
    cases = select_pilot_cases(load_evaluation_cases(EVALUATION_PATH))
    client = QdrantClient(url=os.environ["QDRANT_URL"])
    rows = []

    try:
        for question_id, case in enumerate(cases, start=1):
            query_vector = embed_query(case.question)
            candidates = {}

            for mode in SEARCH_MODES:
                points = search_chunks(
                    client=client,
                    query_text=case.question,
                    query_vector=query_vector,
                    tenant_id="benchmark-tenant",
                    user_id="benchmark-user",
                    limit=10,
                    search_mode=mode,
                    role="member",
                )
                for point in points:
                    payload = point.payload or {}
                    chunk_id = payload.get("chunk_id")
                    if not chunk_id:
                        continue

                    candidate = candidates.setdefault(
                        chunk_id,
                        {
                            "question_id": question_id,
                            "category": case.category,
                            "question": case.question,
                            "chunk_id": chunk_id,
                            "document_id": payload.get("document_id", ""),
                            "location": payload.get("location", ""),
                            "excerpt": " ".join(payload.get("text", "").split())[:220],
                            "retrieval_modes": [],
                            "gold_chunk": chunk_id in case.relevant_chunk_ids,
                            "review_decision": "",
                        },
                    )
                    candidate["retrieval_modes"].append(mode)

            rows.extend(candidates.values())
    finally:
        client.close()

    return rows


def main():
    rows = collect_review_rows()
    if not rows:
        raise RuntimeError("No review candidates were found")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "question_id",
        "category",
        "question",
        "chunk_id",
        "document_id",
        "location",
        "excerpt",
        "retrieval_modes",
        "gold_chunk",
        "review_decision",
    ]
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"review_rows={len(rows)}")
    print(f"gold_chunk_rows={sum(row['gold_chunk'] for row in rows)}")
    print(f"output={OUTPUT_PATH}")


if __name__ == "__main__":
    main()
