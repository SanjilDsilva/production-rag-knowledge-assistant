import json
import logging
from pathlib import Path
import os
import time
from production_rag.providers import create_answer_providers, generate_with_retries


class TimingFileHandler(logging.Handler):
    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self.call_id = 0
        self.run_id = time.time_ns()

    def emit(self, record):
        if record.getMessage() not in {
            "provider_call_succeeded",
            "provider_call_failed",
        }:
            return

        self.call_id += 1
        status = getattr(record, "status_code", None)
        row = {
            "call_id": self.call_id,
            "provider_id": str(getattr(record, "provider", "unknown")),
            "model_id": str(getattr(record, "model", "unknown")),
            "attempt": int(getattr(record, "attempt", 0)),
            "elapsed_ms": float(getattr(record, "elapsed_ms", 0)),
            "status_code": status if isinstance(status, int) else 0,
            "input_tokens": int(getattr(record, "input_tokens", None) or 0),
            "output_tokens": int(getattr(record, "output_tokens", None) or 0),
            "thinking_tokens": int(getattr(record, "thinking_tokens", None) or 0),
            "run_id": self.run_id,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(row) + "\n")


def main():
    timing_handler = TimingFileHandler(Path("data/provider_call_timings.jsonl"))
    provider_logger = logging.getLogger("production_rag.providers")
    provider_logger.setLevel(logging.INFO)
    provider_logger.addHandler(timing_handler)

    primary, backup = create_answer_providers(
        os.environ["GOOGLE_API_KEY"],
        os.environ["GROQ_API_KEY"],
    )
    prompt = [
        {"role": "system", "content": "Reply briefly."},
        {"role": "user", "content": "Reply with exactly OK."},
    ]

    try:
        for provider in (primary, backup):
            for call_number in range(1, 3):
                try:
                    generate_with_retries(
                        provider,
                        prompt,
                        attempts=1,
                        initial_delay_seconds=1,
                        max_delay_seconds=4,
                    )
                except Exception as error:
                    outcome = f"failed ({type(error).__name__})"
                else:
                    outcome = "succeeded"
                print(f"{type(provider).__name__} call {call_number}: {outcome}")
                time.sleep(1)
    finally:
        provider_logger.removeHandler(timing_handler)


if __name__ == "__main__":
    main()
