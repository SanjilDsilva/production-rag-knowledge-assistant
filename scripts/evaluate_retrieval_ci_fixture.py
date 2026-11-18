from pathlib import Path

from qdrant_client import QdrantClient
from production_rag.benchmark_data import (
    import_benchmark_artifacts,
    sha256_file,
    verify_benchmark_manifest,
)
from production_rag.evaluation import load_evaluation_cases, run_retrieval_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_ci"
EVALUATION_PATH = PROJECT_ROOT / "tests" / "fixtures" / "retrieval_eval.jsonl"
METRICS = ("recall@5", "mrr@5", "ndcg@5")
ALLOWED_DROP = 0.02
BASELINE_SCORES = {
    "dense": (0.8156028369, 0.8333333333, 0.7932591243),
    "keyword": (0.9929078014, 0.9468085106, 0.9531548686),
    "hybrid": (0.9751773050, 0.9007092199, 0.9026094717),
}


def main() -> None:
    manifest = verify_benchmark_manifest(FIXTURE_DIR)
    if sha256_file(EVALUATION_PATH) != manifest["evaluation_file_sha256"]:
        raise ValueError("Evaluation questions do not match the fixture manifest")

    cases = load_evaluation_cases(EVALUATION_PATH)
    client = QdrantClient(":memory:")
    try:
        imported = import_benchmark_artifacts(client, FIXTURE_DIR)
        print(f"loaded_questions={len(cases)}")
        print(f"fixture_chunks={imported}")
        for mode in ("dense", "keyword", "hybrid"):
            scores = run_retrieval_evaluation(
                client=client,
                cases=cases,
                tenant_id="benchmark-tenant",
                user_id="benchmark-user",
                k=5,
                search_mode=mode,
            )
            minimums = {
                metric: baseline - ALLOWED_DROP
                for metric, baseline in zip(METRICS, BASELINE_SCORES[mode])
            }
            failures = [
                metric for metric in METRICS if scores[metric] < minimums[metric]
            ]
            print(f"{mode}: scores={scores}; minimums={minimums}")

            if failures:
                print(f"{mode} failed metrics: {failures}; question details:")
                run_retrieval_evaluation(
                    client=client,
                    cases=cases,
                    tenant_id="benchmark-tenant",
                    user_id="benchmark-user",
                    k=5,
                    search_mode=mode,
                    verbose=True,
                )
                raise SystemExit(1)
    finally:
        client.close()


if __name__ == "__main__":
    main()