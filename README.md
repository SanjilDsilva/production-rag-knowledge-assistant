## Retrieval ablation

Measured on the local Qdrant benchmark collection with 10,792 CVE chunks and a 50-question evaluation set. The 3 unanswerable questions are excluded from retrieval scores.

| Search mode | Recall@5 | MRR@5 | nDCG@5 | p50 search latency (ms) | p95 search latency (ms) |
|---|---:|---:|---:|---:|---:|
| Dense | 0.816 | 0.833 | 0.793 | 254.43 | 287.52 |
| BM25 keyword | 0.993 | 0.979 | 0.979 | 783.17 | 949.26 |
| Hybrid (dense + BM25 with RRF) | 0.993 | 0.933 | 0.928 | 1316.79 | 1493.12 |
| Hybrid + reranker | Not evaluated | Not evaluated | Not evaluated | Not measured | Not measured |

Latency measures local Qdrant search calls only; query embedding is excluded. Each mode used 46 warmed samples after one warm-up call. These timings are not API or end-to-end latency and do not establish the app's 2.5-second target.