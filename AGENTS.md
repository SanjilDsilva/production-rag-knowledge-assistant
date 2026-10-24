# AGENTS.md: Enterprise Knowledge Assistant

## Who I am
I am a fresher building a portfolio project for a Python GenAI Engineer job (Capgemini). I know coding basics but cannot yet write code from a blank file. I must be able to explain and defend every line and every design decision in an interview. Use simple words. Whenever you use a technical term, define it in one plain sentence.

## What we are building
An upload-first document question-answering system:
- Users upload their own documents (PDF, text, Word, CSV).
- The system reads, cleans and indexes them in the background and shows a status (queued, processing, ready, failed).
- Users ask questions and get answers with verifiable citations (document name + page/section) in under 2.5 seconds (p95).
- Each user can only see and search their own documents.
- Users can delete a document, and its pieces must disappear from search too.
- The system says "I can't find that in your documents" when the sources do not support an answer.
The NVD CVE records and NIST PDFs are NOT the product. They are test and benchmark data uploaded through the same pipeline, to prove the system handles 10,000 documents and the latency target.

## Job requirements to cover
Python + SQL, FastAPI, LLM app with prompt engineering and embeddings, RAG with LangChain and/or LlamaIndex, vector database, ETL/data pipeline, Docker deployment (README explains how it maps to AWS/Azure/GCP), LLMOps (CI/CD, monitoring, versioning of prompts/data/index, evals, basic security).

## Budget rules
- Default is $0: free tiers only (Gemini free tier with Groq/Cerebras fallback, open-source embedding models, Chroma/FAISS/Qdrant local, SQLite, GitHub Actions on a public repo, Hugging Face Spaces with Docker).
- Absolute maximum is $10 total, only if a free option cannot satisfy a job requirement. Before ANY paid or card-required step: stop, explain what and why, give the estimated cost and a teardown step, then wait for my approval.
- Never create cloud resources, enter payment details, or enable billing. I was burned once by an unpaid cloud bill.
- Free servers lose files when they restart. Tell me about this limit and make a decision record for how uploads are stored.

## Where things run
- VS Code (this local repo, Codex): all application code, tests, Dockerfile, CI, README.
- Google Colab (free GPU): ONLY the one-off heavy job of embedding the 10,000 benchmark documents. Keep the notebook in notebooks/. Walk me through it step by step: open from GitHub, choose a GPU runtime, run, and save the result somewhere the app can load it.
- The deployed app embeds user uploads itself on a small CPU-friendly model.
- No API keys or tokens in notebooks or committed files. Use .env locally (gitignored), Colab secrets in Colab, and Space secrets when deployed.

## Python tooling
Use uv for everything (uv add, uv sync, uv run). Never run pip install and never install packages globally. Before adding any package, tell me what it does in one plain sentence and wait for my approval.

## Targets (definition of done)
- Hybrid search: meaning-based (vector) search plus exact-word (BM25) search, combined with Reciprocal Rank Fusion. Metadata filtering supported.
- Optional reranker, kept only if measurements show it helps enough to justify its delay.
- Ablation table in the README: vector-only vs word-only vs hybrid vs hybrid+reranker, with recall@k, MRR, nDCG and p50/p95 latency.
- Citations built from retrieved chunk metadata, never from the AI's memory. A check after generation verifies every citation was really retrieved.
- Treat document text as untrusted. Defend against instructions hidden inside documents (prompt injection) and test it.
- Handle messy files: corrupt PDFs, empty files, bad encodings are logged and skipped, never silently dropped or crashing.
- Safe uploads: file size limits, allowed types only, scanning for obviously bad files.
- Background processing with status, duplicates detected by content hash, new versions of a file handled, delete supported.
- Per-user separation, with a test proving user A cannot see user B's documents.
- Latency: p95 under 2.5s. Provide a latency breakdown (embedding, search, fusion, rerank, AI model, citation check). Caching with reported hit rate. Retries with backoff and automatic fallback between free AI providers. Friendly message when all free quotas run out. Free AI limits break naive load tests, so measure search speed under load (Locust) separately from full-answer speed at low load, and say clearly which is which.
- Evaluation: at least 50 labeled questions (exact-word, paraphrased, multi-document and unanswerable questions). Runs in CI as a regression gate.
- Production basics: typed API (Pydantic), /health, /ready, /metrics, /version, request ids, structured logs (latency, errors, tokens, cache hits, cost estimate), rate limiting, secrets only in env, slim non-root Docker image, CI/CD with lint, type check, tests, eval gate, dependency scan, and auto-deploy to Hugging Face Spaces.
- Portfolio-ready: clickable sample questions, quota-exhausted message, note that free Spaces may cold-start, architecture diagram, measured numbers, cost per 1,000 queries estimate, NVD non-endorsement notice and NIST attribution, any sample/test data clearly labeled synthetic.
- docs/decisions/: short decision records (ADR-001 onward) for each major choice, listing options and tradeoffs.
- At the end: docs/interview-prep.md with the 15 hardest reviewer questions and answer outlines based on what we actually built.

## LEARNING LADDER (how you must teach me code)
For every piece of core logic, follow these steps and wait for me after each one:
1. Explain the idea in plain words with a tiny real-life example.
2. Show a very small example (under 15 lines) and explain it line by line.
3. Give me a version with blanks (TODO marks) for me to fill in. I try first.
4. I write the real version myself, using the example only as a hint.
5. Review my code like a teammate's pull request: find bugs first, explain what went wrong in simple words, then ask me 2-3 senior-engineer questions (what happens on API timeout? why this chunk size? what breaks at 100k documents? where does the time go?).
Never write the final core code for me. If I am stuck for a long time, give a smaller hint, not the full answer. You may write boilerplate (folders, config, tests scaffolding, Dockerfile, CI files), but explain it afterwards in simple words.
For each major design decision, make me compare two options and their tradeoffs before choosing. Quiz me briefly before each milestone ends.

## Scope guardrails
- Work on ONE milestone at a time, then STOP and wait for me.
- Do not add features, frameworks or refactors beyond the current milestone. If something seems missing, propose it and wait.
- Do not run git push, delete files, install packages, or run anything that costs money without asking. Show me commands before anything destructive.
- Do not download large data without asking.

## Milestones (target about 3 weeks, quality first)
If a milestone needs more time, tell me honestly and we extend it. Never cut scope silently or skip a requirement.
1. Foundation + reading files: project setup, loaders for PDF/text/Word/CSV, cleaning, handling bad files.
2. Chunking + indexing: cutting documents into pieces with metadata, embeddings, keyword index, content-hash dedupe. Colab walkthrough for bulk embedding.
3. Hybrid search + citations + evaluation: vector + BM25 + fusion, citation building and verification, eval set, ablation table.
4. Upload system: FastAPI upload endpoint, background processing with status, per-user separation, delete, file limits and safety.
5. Speed: profiling, caching, retries, fallback, load test with the 10,000-document benchmark.
6. Ship it: Docker, CI/CD with eval gate, deploy to Hugging Face Spaces. (I apply for the job once it is live.)
7. Polish: monitoring, versioning, security checks, README, diagram, decision records, interview-prep doc, retro, resume bullets with real measured numbers.