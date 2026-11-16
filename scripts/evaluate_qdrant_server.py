import os
from pathlib import Path

from qdrant_client import QdrantClient

from production_rag.evaluation import (
    load_evaluation_cases,
    run_retrieval_evaluation,
)

client = QdrantClient(url=os.environ["QDRANT_URL"])
cases = load_evaluation_cases(Path("tests/fixtures/retrieval_eval.jsonl"))

for mode in ("dense", "keyword", "hybrid"):
    scores = run_retrieval_evaluation(
        client=client,
        cases=cases,
        tenant_id="benchmark-tenant",
        user_id="benchmark-user",
        k=5,
        search_mode=mode,
    )
    print(f"{mode}: {scores}")

client.close()
