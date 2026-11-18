from dataclasses import dataclass
from typing import Protocol
from google import genai
from google.genai import types
from groq import Groq
import logging
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from production_rag.settings import load_llm_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    thinking_tokens: int | None = None
    finish_reason: str | None = None
    model_name: str | None = None


class AnswerProvider(Protocol):
    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        raise NotImplementedError


@dataclass(frozen=True)
class RequestDeadline:
    expires_at: float

    @classmethod
    def after(cls, seconds: float) -> "RequestDeadline":
        return cls(time.monotonic() + max(0.0, seconds))

    def remaining_seconds(self) -> float:
        return max(0.0, self.expires_at - time.monotonic())


class GeminiAnswerProvider:
    def __init__(
        self,
        client: genai.Client,
        model_name: str,
        max_output_tokens: int,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.thinking_level: str | None = None

    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        return self._generate_answer(prompt, timeout_seconds=None)

    def generate_answer_with_timeout(
        self,
        prompt: list[dict[str, str]],
        timeout_seconds: float,
    ) -> ProviderResult:
        return self._generate_answer(prompt, timeout_seconds)

    def _generate_answer(
        self,
        prompt: list[dict[str, str]],
        timeout_seconds: float | None,
    ) -> ProviderResult:
        if (
            len(prompt) != 2
            or prompt[0].get("role") != "system"
            or prompt[1].get("role") != "user"
        ):
            raise ValueError("Prompt must have a system message and a user message")
        request_config = types.GenerateContentConfig(
            system_instruction=prompt[0]["content"],
            max_output_tokens=self.max_output_tokens,
            temperature=0.0,
            thinking_config=(
                types.ThinkingConfig(thinking_level=self.thinking_level)
                if self.thinking_level
                else None
            ),
            http_options=(
                types.HttpOptions(timeout=max(1, int(timeout_seconds * 1000)))
                if timeout_seconds is not None
                else None
            ),
        )
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt[1]["content"],
            config=request_config,
        )

        answer = response.text or ""
        if not answer.strip():
            raise RuntimeError("Gemini returned an empty answer")

        usage = response.usage_metadata
        candidates = getattr(response, "candidates", None) or []
        candidate = candidates[0] if candidates else None

        finish_reason = str(candidate.finish_reason) if candidate else None

        return ProviderResult(
            text=answer.strip(),
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
            thinking_tokens=(
                getattr(usage, "thoughts_token_count", None) if usage else None
            ),
            finish_reason=finish_reason,
            model_name=self.model_name,
        )


class GroqAnswerProvider:
    def __init__(
        self,
        client: Groq,
        model_name: str,
        max_output_tokens: int,
    ) -> None:
        self.client = client
        self.model_name = model_name
        self.max_output_tokens = max_output_tokens
        self.reasoning_effort: str | None = None

    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        return self._generate_answer(prompt, timeout_seconds=None)

    def generate_answer_with_timeout(
        self,
        prompt: list[dict[str, str]],
        timeout_seconds: float,
    ) -> ProviderResult:
        return self._generate_answer(prompt, timeout_seconds)

    def _generate_answer(
        self,
        prompt: list[dict[str, str]],
        timeout_seconds: float | None,
    ) -> ProviderResult:
        request_options: dict[str, object] = {
            "model": self.model_name,
            "messages": prompt,
            "temperature": 0.0,
            "max_completion_tokens": self.max_output_tokens,
        }
        if self.reasoning_effort is not None:
            request_options["reasoning_effort"] = self.reasoning_effort

        client = (
            self.client.with_options(timeout=timeout_seconds)
            if timeout_seconds is not None
            else self.client
        )
        response = client.chat.completions.create(**request_options)

        if not response.choices:
            raise RuntimeError("Groq returned no answer choices")

        choice = response.choices[0]
        answer = choice.message.content or ""
        if not answer.strip():
            raise RuntimeError("Groq returned an empty answer")

        usage = response.usage
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        finish = getattr(choice, "finish_reason", None)

        return ProviderResult(
            text=answer.strip(),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
            thinking_tokens=(
                getattr(details, "reasoning_tokens", None) if details else None
            ),
            finish_reason=str(finish) if finish is not None else None,
            model_name=self.model_name,
        )


TRANSIENT_HTTP_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


def safe_provider_error_message(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)

    if status_code == 429:
        return "provider rate limit or quota response"
    if isinstance(status_code, int) and status_code >= 500:
        return "provider server error"
    if isinstance(error, TimeoutError):
        return "provider request timed out"
    if isinstance(error, ConnectionError):
        return "provider connection failed"
    if isinstance(error, RuntimeError):
        return "provider returned an unusable response"

    return "provider request failed; raw message omitted"


def retry_after_seconds(error: Exception) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after")
    if value is None:
        return None

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None

        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        return max(
            0.0,
            (retry_at - datetime.now(timezone.utc)).total_seconds(),
        )


def safe_rate_limit_headers(error: Exception) -> dict[str, str]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", {}) or {}
    allowed_names = {
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    }
    return {
        name.lower(): str(value)
        for name, value in headers.items()
        if name.lower() in allowed_names
    }


@dataclass
class RetryWaitBudget:
    """Track retry sleep time shared by every provider call in one answer."""

    max_wait_seconds: float | None
    waited_seconds: float = 0.0

    def reserve(self, seconds: float) -> bool:
        if (
            self.max_wait_seconds is not None
            and self.waited_seconds + seconds > self.max_wait_seconds
        ):
            return False
        self.waited_seconds += seconds
        return True


def is_temporary_provider_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if status_code is None:
        status_code = getattr(error, "code", None)

    if status_code in TRANSIENT_HTTP_STATUS_CODES:
        return True

    error_name = type(error).__name__.casefold()
    return "timeout" in error_name or "connection" in error_name


def generate_with_retries(
    provider: AnswerProvider,
    prompt: list[dict[str, str]],
    attempts: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
    retry_wait_budget: RetryWaitBudget | None = None,
    wait_stage: str = "model_generation",
    request_deadline: RequestDeadline | None = None,
) -> ProviderResult:
    if attempts < 1:
        raise ValueError("Attempts must be at least 1")

    delay = min(initial_delay_seconds, max_delay_seconds)
    if retry_wait_budget is None:
        retry_wait_budget = RetryWaitBudget(max_wait_seconds=None)

    for attempt in range(1, attempts + 1):
        started_at = time.perf_counter()

        try:
            remaining_seconds = (
                request_deadline.remaining_seconds()
                if request_deadline is not None
                else None
            )
            if remaining_seconds is not None and remaining_seconds <= 0:
                raise RequestDeadlineExceeded("Request deadline expired")
            timed_generate = getattr(provider, "generate_answer_with_timeout", None)
            if remaining_seconds is not None and timed_generate is not None:
                result = timed_generate(prompt, remaining_seconds)
            else:
                result = provider.generate_answer(prompt)
            if (
                request_deadline is not None
                and request_deadline.remaining_seconds() <= 0
            ):
                raise RequestDeadlineExceeded("Request deadline expired")
        except RequestDeadlineExceeded:
            raise
        except Exception as error:
            if (
                request_deadline is not None
                and request_deadline.remaining_seconds() <= 0
            ):
                raise RequestDeadlineExceeded("Request deadline expired") from error
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            status_code = getattr(error, "status_code", None)
            if status_code is None:
                status_code = getattr(error, "code", None)
            retry_after = retry_after_seconds(error)
            logger.warning(
                "provider_call_failed",
                extra={
                    "provider": type(provider).__name__,
                    "model": getattr(provider, "model_name", "unknown"),
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "status_code": status_code,
                    "error_type": type(error).__name__,
                    "error_message": safe_provider_error_message(error),
                    "rate_limit_failure": status_code == 429,
                    "finish_reason": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "thinking_tokens": None,
                    "retry_after_seconds": retry_after,
                    "rate_limit_headers": safe_rate_limit_headers(error),
                },
            )
            if not is_temporary_provider_error(error) or attempt == attempts:
                raise
            wait_seconds = delay if retry_after is None else retry_after
            if (
                request_deadline is not None
                and wait_seconds >= request_deadline.remaining_seconds()
            ):
                logger.warning(
                    "provider_retry_wait_skipped",
                    extra={
                        "provider": type(provider).__name__,
                        "attempt": attempt,
                        "requested_wait_seconds": wait_seconds,
                        "reason": "request_deadline",
                        "stage": wait_stage,
                    },
                )
                raise
            if not retry_wait_budget.reserve(wait_seconds):
                logger.warning(
                    "provider_retry_wait_skipped",
                    extra={
                        "provider": type(provider).__name__,
                        "attempt": attempt,
                        "status_code": status_code,
                        "retry_after_seconds": retry_after,
                        "requested_wait_seconds": wait_seconds,
                        "remaining_wait_budget_seconds": max(
                            0.0,
                            retry_wait_budget.max_wait_seconds
                            - retry_wait_budget.waited_seconds,
                        ),
                        "stage": wait_stage,
                    },
                )
                raise
            logger.info(
                "provider_retry_wait",
                extra={
                    "provider": type(provider).__name__,
                    "attempt": attempt,
                    "status_code": status_code,
                    "retry_after_seconds": retry_after,
                    "wait_seconds": wait_seconds,
                },
            )
            wait_started_at = time.perf_counter()
            time.sleep(wait_seconds)
            logger.info(
                "provider_retry_wait_completed",
                extra={
                    "provider": type(provider).__name__,
                    "attempt": attempt,
                    "status_code": status_code,
                    "retry_after_seconds": retry_after,
                    "requested_wait_seconds": wait_seconds,
                    "elapsed_ms": round(
                        (time.perf_counter() - wait_started_at) * 1000,
                        2,
                    ),
                    "stage": wait_stage,
                },
            )
            delay = min(delay * 2, max_delay_seconds)
        else:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "provider_call_succeeded",
                extra={
                    "provider": type(provider).__name__,
                    "model": result.model_name
                    or getattr(provider, "model_name", "unknown"),
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "status_code": 200,
                    "error_type": None,
                    "error_message": None,
                    "rate_limit_failure": False,
                    "finish_reason": result.finish_reason,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "thinking_tokens": result.thinking_tokens,
                },
            )
            return result

    raise RuntimeError("Provider attempts ended without a result")


class ProviderUnavailableError(RuntimeError):
    pass


class RequestDeadlineExceeded(ProviderUnavailableError):
    pass


def generate_with_fallback(
    primary: AnswerProvider,
    backup: AnswerProvider,
    prompt: list[dict[str, str]],
    attempts: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
    retry_wait_budget: RetryWaitBudget | None = None,
    wait_stage: str = "model_generation",
    request_deadline: RequestDeadline | None = None,
) -> ProviderResult:
    if retry_wait_budget is None:
        retry_wait_budget = RetryWaitBudget(max_wait_seconds=None)

    for provider in (primary, backup):
        try:
            return generate_with_retries(
                provider,
                prompt,
                attempts,
                initial_delay_seconds,
                max_delay_seconds,
                retry_wait_budget,
                wait_stage,
                request_deadline,
            )
        except RequestDeadlineExceeded as error:
            logger.error(
                "request_deadline_exceeded",
                extra={"provider": type(provider).__name__},
            )
            raise error
        except Exception as error:
            logger.error(
                "provider_exhausted",
                extra={
                    "provider": type(provider).__name__,
                    "error_type": type(error).__name__,
                    "status_code": getattr(
                        error, "status_code", getattr(error, "code", None)
                    ),
                },
            )

    raise ProviderUnavailableError(
        "The answer service is temporarily unavailable. Please try again shortly."
    )


def create_answer_providers(
    gemini_api_key: str,
    groq_api_key: str,
) -> tuple[GeminiAnswerProvider, GroqAnswerProvider]:
    config = load_llm_config()
    timeout_seconds = config["limits"]["request_timeout_seconds"]
    output_limit = config["limits"]["output_tokens"]

    gemini_client = genai.Client(
        api_key=gemini_api_key,
        http_options=types.HttpOptions(
            timeout=timeout_seconds * 1000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )

    groq_client = Groq(
        api_key=groq_api_key,
        timeout=timeout_seconds,
        max_retries=0,
    )

    primary = GeminiAnswerProvider(
        gemini_client,
        config["models"]["primary"],
        output_limit,
    )

    backup = GroqAnswerProvider(
        groq_client,
        config["models"]["backup"],
        output_limit,
    )
    return primary, backup
