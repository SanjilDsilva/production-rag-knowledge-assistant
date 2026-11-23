# ADR-002: Semantic Answer Caching in Qdrant

## Status

Accepted for the current single-worker development and Space deployment.

## Context

Repeated questions with exact phrasing are caught by the SHA-256 exact-match cache (< 5ms). However, rephrased questions (e.g. *"What is the primary fix for Log4j?"* vs *"How can I fix Log4j?"*) previously resulted in a cache miss, subjecting users to upstream LLM latency (1.5s–8s) and consuming free-tier LLM token quotas.

## Decision

Store question embeddings and answers in a dedicated Qdrant collection (`semantic_cache`).

1. **Pipeline placement:** Security check -> Per-user rate limit -> Exact hash cache -> Query embedding (`bge-small-en-v1.5`) -> **Semantic cache lookup** -> Hybrid retrieval -> LLM generation -> Citation validation -> Store into exact and semantic cache.
2. **Pre-retrieval security filter:** Qdrant vector queries are strictly filtered by `tenant_id`, `user_id`, `document_set_version`, `prompt_version`, and `model_name`. Cross-tenant or cross-user leakage is impossible at the database level.
3. **Calibrated similarity threshold:** Calibrated against `BAAI/bge-small-en-v1.5` embeddings on paraphrase vs near-miss test pairs:
   - Paraphrase mean: 0.9181 (e.g., Log4j fix: 0.9300, candidate details: 0.9611).
   - Near-miss mean: 0.8190 (e.g., name vs email: 0.8751).
   - Threshold chosen: **0.92**.
4. **Identifier verification:** In addition to cosine similarity $\ge 0.92$, queries must share identical extracted identifiers (CVE IDs, version numbers, capitalized proper nouns). For example, CVE-2021-44228 vs CVE-2022-22965 had high cosine similarity (0.9201), but the identifier check blocks false hits with 100% precision.
5. **Post-hit validity check:** On a cache hit, the pipeline checks whether all cited document IDs are still in `ready_document_ids`. If any document was deleted, the cache entry is treated as a miss.
6. **Eviction and invalidation:** Document uploads or deletions bump `document_set_version`, making old entries logically unreachable. Deletion also triggers `purge_user_semantic_cache`.

## Consequences and tradeoffs

- **Advantages:** Sub-20ms responses for rephrased questions; zero additional third-party dependencies ($0 budget); protects free-tier LLM burst quotas; pre-retrieval tenant isolation.
- **Tradeoffs & local mode behavior:** In Qdrant local embedded mode (`path=...`), single-process disk locking applies; this is fine for single-container Hugging Face Spaces. In a multi-worker server environment, Qdrant server mode (`url=...`) provides concurrent read/write scaling.
