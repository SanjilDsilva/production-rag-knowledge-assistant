FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    RAG_DATA_DIR=/app/data

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY src ./src
COPY Config ./Config

RUN groupadd --system app \
    && useradd --system --create-home --gid app app \
    && mkdir -p /app/data \
    && chown -R app:app /app /home/app

USER app
EXPOSE 8000

CMD ["/app/.venv/bin/uvicorn", "production_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]