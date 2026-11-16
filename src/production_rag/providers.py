from dataclasses import dataclass
from typing import Protocol
from google import genai
from google.genai import types
from groq import Groq
import logging
import time

from production_rag.settings import load_llm_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProviderResult:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class AnswerProvider(Protocol):
    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        raise NotImplementedError


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

    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        if (
            len(prompt) != 2
            or prompt[0].get("role") != "system"
            or prompt[1].get("role") != "user"
        ):
            raise ValueError("Prompt must have a system message and a user message")
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt[1]["content"],
            config=types.GenerateContentConfig(
                system_instruction=prompt[0]["content"],
                max_output_tokens=self.max_output_tokens,
                temperature=0.0,
            ),
        )

        answer = response.text or ""
        if not answer.strip():
            raise RuntimeError("Gemini returned an empty answer")

        usage = response.usage_metadata
        return ProviderResult(
            text=answer.strip(),
            input_tokens=usage.prompt_token_count if usage else None,
            output_tokens=usage.candidates_token_count if usage else None,
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

    def generate_answer(self, prompt: list[dict[str, str]]) -> ProviderResult:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=prompt,
            temperature=0.0,
            max_completion_tokens=self.max_output_tokens,
        )

        if not response.choices:
            raise RuntimeError("Groq returned no answer choices")

        answer = response.choices[0].message.content or ""
        if not answer.strip():
            raise RuntimeError("Groq returned an empty answer")

        usage = response.usage
        return ProviderResult(
            text=answer.strip(),
            input_tokens=usage.prompt_tokens if usage else None,
            output_tokens=usage.completion_tokens if usage else None,
        )


TRANSIENT_HTTP_STATUS_CODES = {
    408,
    429,
    500,
    502,
    503,
    504,
}


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
) -> ProviderResult:
    if attempts < 1:
        raise ValueError("Attempts must be at least 1")

    delay = min(initial_delay_seconds, max_delay_seconds)

    for attempt in range(1, attempts + 1):
        started_at = time.perf_counter()

        try:
            result = provider.generate_answer(prompt)
        except Exception as error:
            elapsed_ms = (time.perf_counter() - started_at) * 1000

            logger.warning(
                "provider_call_failed",
                extra={
                    "provider": type(provider).__name__,
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "error_type": type(error).__name__,
                    "status_code": getattr(
                        error, "status_code", getattr(error, "code", None)
                    ),
                },
            )
            if not is_temporary_provider_error(error) or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, max_delay_seconds)
        else:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "provider_call_succeeded",
                extra={
                    "provider": type(provider).__name__,
                    "attempt": attempt,
                    "elapsed_ms": round(elapsed_ms, 2),
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                },
            )
            return result

    raise RuntimeError("Provider attempts ended without a result")


class ProviderUnavailableError(RuntimeError):
    pass


def generate_with_fallback(
    primary: AnswerProvider,
    backup: AnswerProvider,
    prompt: list[dict[str, str]],
    attempts: int,
    initial_delay_seconds: float,
    max_delay_seconds: float,
) -> ProviderResult:
    for provider in (primary, backup):
        try:
            return generate_with_retries(
                provider,
                prompt,
                attempts,
                initial_delay_seconds,
                max_delay_seconds,
            )
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
