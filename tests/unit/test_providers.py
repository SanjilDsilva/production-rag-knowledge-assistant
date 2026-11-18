from production_rag.providers import (
    AnswerProvider,
    GeminiAnswerProvider,
    GroqAnswerProvider,
    ProviderResult,
    is_temporary_provider_error,
    generate_with_retries,
    generate_with_fallback,
    ProviderUnavailableError,
    RetryWaitBudget,
    create_answer_providers,
    safe_rate_limit_headers,
)
from types import SimpleNamespace
import logging
import pytest

TEST_PROMPT = [
    {"role": "system", "content": "Use only retrieved sources."},
    {"role": "user", "content": "What is a test?"},
]


class FakeProvider:
    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        return ProviderResult(
            text=f"Fake answer to: {prompt[1]['content']}",
            input_tokens=8,
            output_tokens=4,
        )


def test_fake_provider_uses_shared_answer_shape():
    provider: AnswerProvider = FakeProvider()

    result = provider.generate_answer(TEST_PROMPT)

    assert result.text == "Fake answer to: What is a test?"
    assert result.input_tokens == 8
    assert result.output_tokens == 4


class FakeGeminiClient:
    def __init__(self):
        self.request = None
        self.models = SimpleNamespace(generate_content=self.generate_content)

    def generate_content(self, **request):
        self.request = request
        return SimpleNamespace(
            text=" Supported answer [chunk-1] ",
            usage_metadata=SimpleNamespace(
                prompt_token_count=12,
                candidates_token_count=5,
            ),
        )


def test_gemini_provider_returns_answer_and_token_counts():
    client = FakeGeminiClient()
    provider = GeminiAnswerProvider(
        client=client,
        model_name="test-model",
        max_output_tokens=64,
    )

    result = provider.generate_answer(TEST_PROMPT)

    assert result.text == "Supported answer [chunk-1]"
    assert result.input_tokens == 12
    assert result.output_tokens == 5
    assert client.request["model"] == "test-model"
    assert client.request["config"].max_output_tokens == 64
    assert client.request["contents"] == "What is a test?"
    assert client.request["config"].system_instruction == "Use only retrieved sources."


def test_gemini_provider_uses_remaining_request_timeout():
    client = FakeGeminiClient()
    provider = GeminiAnswerProvider(client, "test-model", 64)

    provider.generate_answer_with_timeout(TEST_PROMPT, 2.5)

    assert client.request["config"].http_options.timeout == 2500


class FakeGroqProviderClient:
    def __init__(self):
        self.request = None
        self.timeout = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def with_options(self, **options):
        self.timeout = options["timeout"]
        return self

    def create(self, **request):
        self.request = request
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=" Supported answer [chunk-1] ")
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=14,
                completion_tokens=6,
            ),
        )


def test_groq_provider_returns_answer_and_token_counts():
    client = FakeGroqProviderClient()
    provider = GroqAnswerProvider(
        client=client,
        model_name="test-model",
        max_output_tokens=64,
    )

    result = provider.generate_answer(TEST_PROMPT)

    assert result.text == "Supported answer [chunk-1]"
    assert result.input_tokens == 14
    assert result.output_tokens == 6
    assert client.request["model"] == "test-model"
    assert client.request["max_completion_tokens"] == 64
    assert client.request["messages"] == TEST_PROMPT


def test_groq_provider_uses_remaining_request_timeout():
    client = FakeGroqProviderClient()
    provider = GroqAnswerProvider(client, "test-model", 64)

    provider.generate_answer_with_timeout(TEST_PROMPT, 2.5)

    assert client.timeout == 2.5


class FakeRateLimitError(Exception):
    status_code = 429


def test_safe_rate_limit_headers_keeps_only_allowed_headers():
    error = FakeRateLimitError("too many requests")
    error.response = SimpleNamespace(
        headers={
            "Retry-After": "7",
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-limit-tokens": "6000",
            "authorization": "do-not-log",
        }
    )

    assert safe_rate_limit_headers(error) == {
        "retry-after": "7",
        "x-ratelimit-limit-requests": "100",
        "x-ratelimit-limit-tokens": "6000",
    }


def test_timeout_error_is_temporary():
    assert is_temporary_provider_error(TimeoutError("request timed out"))


def test_rate_limit_error_is_temporary():
    error = FakeRateLimitError("too many requests")

    assert is_temporary_provider_error(error)


class FakeSequenceProvider:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = 0

    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_retries_timeout_and_rate_limit_with_growing_waits(monkeypatch):
    waits = []
    monkeypatch.setattr("production_rag.providers.time.sleep", waits.append)
    expected = ProviderResult("Final answer", 3, 2)
    provider = FakeSequenceProvider(
        [TimeoutError("slow"), FakeRateLimitError("busy"), expected]
    )

    result = generate_with_retries(provider, TEST_PROMPT, 3, 1, 4)

    assert result == expected
    assert provider.calls == 3
    assert waits == [1, 2]


def test_retry_after_header_overrides_backoff(monkeypatch):
    waits = []
    monkeypatch.setattr("production_rag.providers.time.sleep", waits.append)

    error = FakeRateLimitError("too many requests")
    error.response = SimpleNamespace(headers={"retry-after": "7"})
    provider = FakeSequenceProvider([error, ProviderResult("Recovered answer")])

    result = generate_with_retries(provider, TEST_PROMPT, 2, 1, 4)

    assert result.text == "Recovered answer"
    assert waits == [7.0]


def test_retry_after_over_budget_skips_sleep_and_uses_backup(monkeypatch):
    waits = []
    monkeypatch.setattr("production_rag.providers.time.sleep", waits.append)
    error = FakeRateLimitError("too many requests")
    error.response = SimpleNamespace(headers={"retry-after": "120"})
    primary = FakeSequenceProvider([error])
    backup = FakeSequenceProvider([ProviderResult("Backup answer")])

    result = generate_with_fallback(
        primary,
        backup,
        TEST_PROMPT,
        attempts=3,
        initial_delay_seconds=1,
        max_delay_seconds=4,
        retry_wait_budget=RetryWaitBudget(max_wait_seconds=10),
    )

    assert result.text == "Backup answer"
    assert waits == []
    assert primary.calls == 1
    assert backup.calls == 1


def test_retry_wait_is_logged_separately_from_provider_call(monkeypatch, caplog):
    waits = []
    monkeypatch.setattr("production_rag.providers.time.sleep", waits.append)
    error = FakeRateLimitError("too many requests")
    error.response = SimpleNamespace(headers={"retry-after": "2"})
    provider = FakeSequenceProvider([error, ProviderResult("Recovered answer")])

    with caplog.at_level(logging.INFO, logger="production_rag.providers"):
        generate_with_retries(
            provider,
            TEST_PROMPT,
            attempts=2,
            initial_delay_seconds=1,
            max_delay_seconds=4,
            retry_wait_budget=RetryWaitBudget(max_wait_seconds=10),
        )

    wait_record = next(
        record for record in caplog.records
        if record.message == "provider_retry_wait_completed"
    )
    call_record = next(
        record for record in caplog.records
        if record.message == "provider_call_succeeded"
    )
    assert waits == [2.0]
    assert wait_record.requested_wait_seconds == 2.0
    assert wait_record.elapsed_ms >= 0
    assert call_record.elapsed_ms >= 0


def test_success_log_has_timing_and_tokens_but_not_prompt(caplog):
    prompt = [
        {"role": "system", "content": "PRIVATE DOCUMENT TEXT"},
        {"role": "user", "content": "Question"},
    ]

    with caplog.at_level(logging.INFO, logger="production_rag.providers"):
        generate_with_retries(FakeProvider(), prompt, 1, 1, 4)

    record = next(
        record
        for record in caplog.records
        if record.message == "provider_call_succeeded"
    )
    assert record.elapsed_ms >= 0
    assert record.input_tokens == 8
    assert record.output_tokens == 4
    assert record.model == "unknown"
    assert record.status_code == 200
    assert record.finish_reason is None
    assert record.thinking_tokens is None
    assert record.rate_limit_failure is False
    assert "PRIVATE DOCUMENT TEXT" not in caplog.text


def test_empty_primary_answer_uses_backup():
    primary = FakeSequenceProvider([RuntimeError("Gemini returned an empty answer")])
    backup = FakeProvider()

    result = generate_with_fallback(primary, backup, TEST_PROMPT, 1, 1, 4)

    assert result.text == "Fake answer to: What is a test?"
    assert primary.calls == 1


def test_both_providers_failing_gives_friendly_error():
    primary = FakeSequenceProvider([TimeoutError("network problem")])
    backup = FakeSequenceProvider([RuntimeError("empty answer")])

    with pytest.raises(
        ProviderUnavailableError,
        match="temporarily unavailable",
    ):
        generate_with_fallback(primary, backup, TEST_PROMPT, 1, 1, 4)

    assert primary.calls == 1
    assert backup.calls == 1


def test_provider_factory_uses_config_and_timeouts(monkeypatch):
    created = {}

    def fake_gemini_client(**options):
        created["gemini"] = options
        return object()

    def fake_groq_client(**options):
        created["groq"] = options
        return object()

    monkeypatch.setattr(
        "production_rag.providers.genai.Client",
        fake_gemini_client,
    )
    monkeypatch.setattr(
        "production_rag.providers.Groq",
        fake_groq_client,
    )

    primary, backup = create_answer_providers("fake-gemini-key", "fake-groq-key")

    assert primary.model_name == "gemini-3.8-flash"
    assert backup.model_name == "openai/gpt-oss-20b"
    assert primary.max_output_tokens == 512
    assert backup.max_output_tokens == 512
    assert created["gemini"]["http_options"].timeout == 15000
    assert created["gemini"]["http_options"].retry_options.attempts == 1
    assert created["groq"]["timeout"] == 15
    assert created["groq"]["max_retries"] == 0


def test_gemini_provider_passes_low_thinking_setting():
    client = FakeGeminiClient()
    provider = GeminiAnswerProvider(client, "test-model", 64)
    provider.thinking_level = "low"

    provider.generate_answer(TEST_PROMPT)

    assert client.request["config"].thinking_config.thinking_level.value == "LOW"


def test_groq_provider_passes_low_reasoning_setting():
    client = FakeGroqProviderClient()
    provider = GroqAnswerProvider(client, "test-model", 64)
    provider.reasoning_effort = "low"

    provider.generate_answer(TEST_PROMPT)

    assert client.request["reasoning_effort"] == "low"
