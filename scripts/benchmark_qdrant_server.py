import math
import os
import time
from pathlib import Path
from statistics import median

from qdrant_client import QdrantClient

from production_rag.embeddings import embed_query
from production_rag.evaluation import load_evaluation_cases
from production_rag.vector_store import search_chunks

client = QdrantClient(url=os.environ["QDRANT_URL"])
cases = load_evaluation_cases(Path("tests/fixtures/retrieval_eval.jsonl"))
cases = [case for case in cases if case.answerable]
prepared = [(case.question, embed_query(case.question)) for case in cases]

for question, vector in prepared:
    search_chunks(
        client, question, vector, "benchmark-tenant", "benchmark-user"
    )

times = []
for _ in range(3):
    for question, vector in prepared:
        started = time.perf_counter()
        search_chunks(
            client, question, vector, "benchmark-tenant", "benchmark-user"
        )
        times.append((time.perf_counter() - started) * 1000)

ordered = sorted(times)
p95 = ordered[math.ceil(0.95 * len(ordered)) - 1]
print(f"unique_questions={len(prepared)} requests={len(times)}")
print(f"search_p50_ms={median(times):.2f} search_p95_ms={p95:.2f}")
client.close()