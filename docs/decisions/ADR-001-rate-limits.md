# ADR-001: In-memory rate limits

## Status

Accepted for the current single-worker development deployment.

## Decision

Use in-memory fixed-window limiters for upload and question protection. The configured defaults are 5 uploads per tenant/user per minute, 2 questions per tenant/user per minute, and global question caps of 3 per minute and 100 per 24-hour window per worker. Reaching a global cap still permits retrieval and returns cited passages in degraded mode without a provider call. The minute cap is based on the measured 8,000 tokens-per-minute provider limit and an estimated 1,500–2,000 tokens per question; a token-aware limiter using remaining-token headers is the planned upgrade.

Count cap-triggered degraded responses separately from provider-failure degraded responses. Successful Groq calls may log only the approved rate-limit header allowlist; never log credentials, prompt text, or answer text.

## Consequences and tradeoffs

- The limiter is small and needs no database or service.
- Counters and windows live in process memory, so they reset when the worker restarts.
- Each worker has its own independent counters. With multiple workers or replicas, the effective deployment-wide cap is multiplied and users may be routed to different counters.
- Fixed windows can allow a burst around a window boundary; stricter smoothing is unnecessary for this small prototype.
- If the app moves to multiple workers or replicas, replace or coordinate the limiter with a shared service such as Redis. That upgrade is not installed or configured now.

## Configuration

Adjust the limits in `Config/llm.toml`. The current global defaults are provisional until measured against the actual provider response headers and supplied-token diagnostics.
