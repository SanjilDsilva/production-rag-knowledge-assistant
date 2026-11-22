#Stage 1: Build the React frontend
FROM node:20-slim AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Stage 2: Production Python runtime (non-root, port 7860 for Hugging Face Spaces)
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /uvx /bin/

# Hugging Face Spaces requires a non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH="/home/user/.local/bin:$PATH" \
    PYTHONPATH=/home/user/app/src \
    PYTHONBUFFERED=1 \
    RAG_DATA_DIR=/home/user/app/data \
    UV_PROJECT_ENVIRONMENT=/home/user/app/.venv \
    UV_COMPILE_BYTECODE=1

WORKDIR /home/user/app

# Copy and install Python dependencies
COPY --chown=user:user pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Copy application source and configurations
COPY --chown=user:user src ./src
COPY --chown=user:user Config ./Config

# Copy compiled React static assets from Stage 1
COPY --from=frontend-builder --chown=user:user /app/frontend/dist ./frontend/dist

# Final project sync and create runtime data directory
RUN uv sync --locked --no-dev && mkdir -p /home/user/app/data

# Render listens on port 1000
EXPOSE 1000

CMD ["/home/user/app/.venv/bin/uvicorn", "production_rag.api:app", "--host", "0.0.0.0", "--port", "1000"]
