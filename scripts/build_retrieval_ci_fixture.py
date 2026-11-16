from collections import Counter
from pathlib import Path
from typing import Any
import json
import numpy as np

from production_rag.benchmark_data import load_benchmark_artifacts
from production_rag.evaluation import EvaluationCase, load_evaluation_cases
from qdrant_client import QdrantClient
from production_rag.benchmark_data import (
    EXPECTED_CHUNKER,
    EXPECTED_MODEL,
    EXPECTED_REVISION,
    import_benchmark_artifacts,
    load_benchmark_artifacts,
    sha256_file,
)
from production_rag.embeddings import embed_query
from production_rag.vector_store import search_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "data" / "benchmark"
EVALUATION_PATH = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval.jsonl"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_ci"


def find_gold_chunks(
    cases: list[EvaluationCase],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gold_ids = {chunk_id for case in cases for chunk_id in case.relevant_chunk_ids}
    records_by_id = {record["chunk_id"]: record for record in records}
    missing_ids = gold_ids - records_by_id.keys()

    if missing_ids:
        raise ValueError(f"Gold chunks missing from benchmark: {len(missing_ids)}")

    return [records_by_id[chunk_id] for chunk_id in sorted(gold_ids)]


def find_hard_distractor_ids(
    client: QdrantClient,
    cases: list[EvaluationCase],
    gold_ids: set[str],
) -> set[str]:
    distractor_ids = set()

    for case in cases:
        query_vector = embed_query(case.question)
        for mode in ("dense", "keyword", "hybrid"):
            results = search_chunks(
                client=client,
                query_text=case.question,
                query_vector=query_vector,
                tenant_id="benchmark-tenant",
                user_id="benchmark-user",
                limit=5,
                search_mode=mode,
            )
            for result in results:
                payload = result.payload or {}
                chunk_id = payload.get("chunk_id")
                if not isinstance(chunk_id, str) or not chunk_id:
                    raise ValueError("Search result is missing its chunk ID")
                if chunk_id not in gold_ids:
                    distractor_ids.add(chunk_id)

    return distractor_ids


def main() -> None:
    vectors, chunks = load_benchmark_artifacts(BENCHMARK_DIR)
    cases = load_evaluation_cases(EVALUATION_PATH)
    gold_chunks = find_gold_chunks(cases, chunks)
    gold_ids = {record["chunk_id"] for record in gold_chunks}
    client = QdrantClient(":memory:")
    try:
        import_benchmark_artifacts(client, BENCHMARK_DIR)
        distractor_ids = find_hard_distractor_ids(client, cases, gold_ids)
    finally:
        client.close()
    selected_ids = gold_ids | distractor_ids
    records_by_id = {record["chunk_id"]: record for record in chunks}
    row_by_id = {record["chunk_id"]: index for index, record in enumerate(chunks)}
    ordered_ids = sorted(selected_ids)
    selected_records = [records_by_id[chunk_id] for chunk_id in ordered_ids]
    selected_vectors = vectors[[row_by_id[chunk_id] for chunk_id in ordered_ids]]

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    chunks_path = FIXTURE_DIR / "chunks.jsonl"
    vectors_path = FIXTURE_DIR / "vectors.npy"
    chunks_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in selected_records
        ),
        encoding="utf-8",
    )
    np.save(vectors_path, selected_vectors)

    manifest = {
        "embedding_model": EXPECTED_MODEL,
        "embedding_model_revision": EXPECTED_REVISION,
        "vector_size": int(vectors.shape[1]),
        "vectors_normalized": True,
        "chunker_version": EXPECTED_CHUNKER,
        "chunk_count": len(selected_records),
        "source_manifest_sha256": sha256_file(BENCHMARK_DIR / "manifest.json"),
        "evaluation_file_sha256": sha256_file(EVALUATION_PATH),
        "files": {
            "vectors.npy": sha256_file(vectors_path),
            "chunks.jsonl": sha256_file(chunks_path),
        },
    }
    (FIXTURE_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    load_benchmark_artifacts(FIXTURE_DIR)
    categories = Counter(case.category for case in cases)

    print(f"benchmark_chunks={len(chunks)}")
    print(f"benchmark_vectors={vectors.shape[0]} x {vectors.shape[1]}")
    print(f"gold_chunks={len(gold_chunks)}")
    print(f"hard_distractors={len(distractor_ids)}")
    print(f"fixture_chunk_candidates={len(gold_ids) + len(distractor_ids)}")
    print(f"evaluation_questions={len(cases)}")
    print(f"evaluation_categories={dict(sorted(categories.items()))}")


if __name__ == "__main__":
    main()
