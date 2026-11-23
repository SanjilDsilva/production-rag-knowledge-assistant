import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager
import hashlib
from threading import Lock
from typing import BinaryIO, Literal
from qdrant_client import QdrantClient
import shutil
from tempfile import TemporaryDirectory
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
from production_rag.chunking import CHUNKER_VERSION

from production_rag.indexing import index_document_file
from production_rag.vector_store import delete_document_chunks

from production_rag.vector_store import COLLECTION_NAME, create_collection
from production_rag.status_store import DocumentStatusStore
from production_rag.storage import LocalUploadStorage, UploadStorage, MAX_UPLOAD_BYTES
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    File,
    UploadFile,
    Form,
    Response,
)
from uuid import uuid4

from production_rag.providers import RequestDeadline, create_answer_providers
from production_rag.settings import load_llm_config
from pydantic import BaseModel, Field
from production_rag.answering import (
    UNSUPPORTED_ANSWER,
    answer_question,
    log_answer_audit,
    PROMPT_VERSION,
)
from production_rag.embeddings import (
    MODEL_NAME,
    MODEL_REVISION,
    is_embedding_model_loaded,
)
from production_rag.query_security import query_block_reason
from production_rag.rate_limit import FixedWindowRateLimiter
from production_rag.answer_cache import (
    BoundedAnswerCache,
    CacheBackend,
    SQLiteCacheBackend,
    create_cache_backend,
)

logger = logging.getLogger(__name__)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=10000)


APP_ENV = os.getenv("APP_ENV", "production").casefold()
DEV_IDENTITY_ENABLED = (
    os.getenv("RAG_DEV_IDENTITY_ENABLED", "false").casefold() == "true"
)

if DEV_IDENTITY_ENABLED and APP_ENV != "development":
    raise RuntimeError("Development identity can only run in development.")


def get_current_identity(
    x_dev_user: str = Header(default="user-a", alias="X_Dev_User"),
) -> dict[str, str]:
    if not DEV_IDENTITY_ENABLED:
        raise HTTPException(status_code=503, detail="Development identity is disabled.")

    dev_identities = {
        # Tenant 1: Acme Corp / Dev Tenant
        "user-a": {
            "tenant_id": "dev-tenant",
            "user_id": "dev-user-a",
            "role": "member",
        },
        "user-b": {
            "tenant_id": "dev-tenant",
            "user_id": "dev-user-b",
            "role": "admin",
        },
        "user-c": {
            "tenant_id": "dev-tenant",
            "user_id": "dev-user-c",
            "role": "member",
        },
        # Tenant 2: Stark Industries
        "user-d": {
            "tenant_id": "stark-industries",
            "user_id": "stark-user-d",
            "role": "admin",
        },
        "user-e": {
            "tenant_id": "stark-industries",
            "user_id": "stark-user-e",
            "role": "member",
        },
        "user-f": {
            "tenant_id": "stark-industries",
            "user_id": "stark-user-f",
            "role": "member",
        },
    }
    if x_dev_user not in dev_identities:
        raise HTTPException(
            status_code=400,
            detail=f"X_Dev_User must be one of: {', '.join(dev_identities.keys())}",
        )
    return dev_identities[x_dev_user]


DATA_ROOT = Path(os.getenv("RAG_DATA_DIR", "data"))


def create_answer_cache() -> CacheBackend[str]:
    return create_cache_backend(
        os.getenv("RAG_CACHE_BACKEND", "memory"),
        namespace="answers",
        max_entries=256,
        ttl_seconds=900,
        database_path=DATA_ROOT / "cache.db",
    )


def create_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    if qdrant_url:
        return QdrantClient(url=qdrant_url)

    return QdrantClient(path=str(DATA_ROOT / "qdrant"))


def initialize_app_storage(app: FastAPI) -> None:
    app.state.upload_storage = LocalUploadStorage(DATA_ROOT / "uploads")
    app.state.status_store = DocumentStatusStore(DATA_ROOT / "app.db")
    app.state.answer_cache = create_answer_cache()
    limits = load_llm_config()["limits"]
    app.state.upload_rate_limiter = FixedWindowRateLimiter(
        max_requests=limits["upload_requests_per_minute"],
        window_seconds=limits["upload_rate_limit_window_seconds"],
    )
    app.state.question_rate_limiter = FixedWindowRateLimiter(
        max_requests=limits["user_questions_per_minute"],
        window_seconds=60,
    )
    app.state.global_question_minute_limiter = FixedWindowRateLimiter(
        max_requests=limits["global_questions_per_minute"],
        window_seconds=60,
    )
    app.state.global_question_daily_limiter = FixedWindowRateLimiter(
        max_requests=limits["global_questions_per_day"],
        window_seconds=86400,
    )
    app.state.degraded_response_counts = {
        "question_cap": 0,
        "provider_failure": 0,
    }
    app.state.degraded_response_lock = Lock()
    app.state.status_store.reset_processing()
    qdrant_client = create_qdrant_client()
    create_collection(qdrant_client)
    app.state.qdrant_client = qdrant_client


def record_degraded_response(app: FastAPI, reason: str | None) -> None:
    if reason in {"global_minute_cap", "global_daily_cap"}:
        category = "question_cap"
    elif reason == "provider_failure":
        category = "provider_failure"
    else:
        return

    with app.state.degraded_response_lock:
        app.state.degraded_response_counts[category] += 1
        count = app.state.degraded_response_counts[category]

    logger.info(
        "degraded_response_count",
        extra={"category": category, "count": count},
    )


def get_upload_storage(request: Request) -> UploadStorage:
    return request.app.state.upload_storage


def get_document_status_store(request: Request) -> DocumentStatusStore:
    return request.app.state.status_store


def hash_upload_file(content: BinaryIO) -> str:
    digest = hashlib.sha256()
    total_bytes = 0

    while True:
        piece = content.read(64 * 1024)
        if not piece:
            break

        total_bytes += len(piece)
        if total_bytes > MAX_UPLOAD_BYTES:
            raise ValueError("File is too large")

        digest.update(piece)

    content.seek(0)
    return digest.hexdigest()


def process_document_background(
    client: QdrantClient,
    storage: UploadStorage,
    status_store: DocumentStatusStore,
    tenant_id: str,
    user_id: str,
    document_id: str,
    original_filename: str,
    storage_key: str,
) -> None:
    started = status_store.update_status(tenant_id, user_id, document_id, "processing")
    if not started:
        return

    document = status_store.get_for_user(tenant_id, user_id, document_id)
    if document is None:
        return
    visibility = document["visibility"]

    extension = Path(original_filename).suffix.casefold()

    try:
        with TemporaryDirectory() as temperory_folder:
            temporary_path = Path(temperory_folder) / f"{document_id} {extension}"
            with storage.read(storage_key) as saved_file:
                with temporary_path.open("wb") as copied_file:
                    shutil.copyfileobj(saved_file, copied_file, length=64 * 1024)

            result = index_document_file(
                client,
                temporary_path,
                tenant_id,
                user_id,
                document_id=document_id,
                original_filename=original_filename,
                visibility=visibility,
            )

        if result.document_id != document_id:
            raise RuntimeError("Indexer returned a different document ID")

    except Exception as error:
        try:
            delete_document_chunks(client, tenant_id, user_id, document_id)
        finally:
            status_store.update_status(
                tenant_id,
                user_id,
                document_id,
                "failed",
                f"Indexing failed ({type(error).__name__})",
            )
        return

    became_ready = status_store.update_status(tenant_id, user_id, document_id, "ready")
    if not became_ready:
        delete_document_chunks(client, tenant_id, user_id, document_id)
        return

    if result.previous_document_id is not None:
        delete_document_chunks(
            client,
            tenant_id,
            user_id,
            result.previous_document_id,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_storage(app)
    try:
        yield
    finally:
        app.state.qdrant_client.close()


PRIVACY_NOTICE = (
    "Privacy: Free-tier AI providers may use submitted text under thier terms. "
    "Only use public or sample documents with free-tier AI providers."
)

app = FastAPI(
    title="Production RAG",
    description=PRIVACY_NOTICE,
    lifespan=lifespan,
)


@app.post("/documents", status_code=202)
def upload_document(
    background_tasks: BackgroundTasks,
    request: Request,
    file: UploadFile = File(...),
    visibility: Literal["private", "tenant_shared"] = Form(default="private"),
    identity: dict[str, str] = Depends(get_current_identity),
    storage: UploadStorage = Depends(get_upload_storage),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, str]:
    limiter = request.app.state.upload_rate_limiter
    if not limiter.allow(identity["tenant_id"], identity["user_id"]):
        wait_seconds = limiter.retry_after_seconds(
            identity["tenant_id"], identity["user_id"]
        )
        raise HTTPException(
            status_code=429,
            detail="Too many uploads. Please wait before trying again.",
            headers={"Retry-After": str(wait_seconds)},
        )
    original_filename = file.filename or "uploaded-file"
    extension = Path(original_filename).suffix.casefold()
    try:
        file_hash = hash_upload_file(file.file)
    except ValueError as error:
        status_code = 413 if "too large" in str(error).casefold() else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error

    existing = status_store.find_active_by_file_hash(
        identity["tenant_id"],
        identity["user_id"],
        file_hash,
    )

    if existing is not None:
        return {
            "document_id": existing["document_id"],
            "filename": existing["original_filename"],
            "status": existing["status"],
        }
    document_id = str(uuid4())

    try:
        storage_key = storage.save(
            identity["tenant_id"],
            document_id,
            extension,
            file.file,
        )
    except ValueError as error:
        status_code = 413 if "too large" in str(error).casefold() else 400
        raise HTTPException(status_code=status_code, detail=str(error)) from error

    try:
        status_store.create_queued(
            identity["tenant_id"],
            identity["user_id"],
            document_id,
            original_filename,
            file_hash,
            storage_key,
            visibility=visibility,
        )
    except Exception:
        storage.delete(storage_key)
        raise

    background_tasks.add_task(
        process_document_background,
        request.app.state.qdrant_client,
        storage,
        status_store,
        identity["tenant_id"],
        identity["user_id"],
        document_id,
        original_filename,
        storage_key,
    )

    return {
        "document_id": document_id,
        "filename": original_filename,
        "status": "queued",
    }


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def readiness_endpoint(request: Request, response: Response) -> dict[str, object]:
    checks = {
        "qdrant": "ok",
        "embedding_model": ("loaded" if is_embedding_model_loaded() else "not_loaded"),
    }
    try:
        request.app.state.qdrant_client.get_collections()
    except Exception:
        checks["qdrant"] = "unavailable"

    ready = checks["qdrant"] == "ok" and checks["embedding_model"] == "loaded"
    response.status_code = 200 if ready else 503
    return {"status": "ready" if ready else "not_ready", "checks": checks}


def get_app_version() -> str:
    project_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with project_file.open("rb") as file:
        project = tomllib.load(file)
    return project["project"]["version"]


@app.get("/version")
def version_endpoint() -> dict[str, str]:
    limits = load_llm_config()["limits"]
    label_mode = (
        "prompt_order"
        if limits.get("citation_labels_follow_prompt_order", True)
        else "chunk_id_order"
    )
    return {
        "app_version": get_app_version(),
        "embedding_model": MODEL_NAME,
        "embedding_model_revision": MODEL_REVISION,
        "prompt_version": PROMPT_VERSION,
        "chunker_version": CHUNKER_VERSION,
        "citation_label_mode": label_mode,
    }


@app.get("/documents")
def list_documents(
    identity: dict[str, str] = Depends(get_current_identity),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, list[dict[str, str | None]]]:
    documents = status_store.list_accessible(
        identity["tenant_id"],
        identity["user_id"],
        identity["role"],
    )
    return {"documents": documents}


@app.delete("/documents/{document_id}")
def delete_document(
    document_id: str,
    request: Request,
    identity: dict[str, str] = Depends(get_current_identity),
    storage: UploadStorage = Depends(get_upload_storage),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, str]:
    document = status_store.get_visible_document(
        identity["tenant_id"],
        identity["user_id"],
        identity["role"],
        document_id,
    )
    if document is None or (
        identity["role"] == "member" and document["user_id"] != identity["user_id"]
    ):
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document_chunks(
        request.app.state.qdrant_client,
        document["tenant_id"],
        document["user_id"],
        document_id,
    )

    storage.delete(document["storage_key"])
    status_store.delete_for_user(
        document["tenant_id"],
        document["user_id"],
        document_id,
    )
    return {"document_id": document_id, "status": "deleted"}


@app.post("/questions")
def ask_question(
    payload: QuestionRequest,
    request: Request,
    identity: dict[str, str] = Depends(get_current_identity),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, object]:
    request_deadline = RequestDeadline.after(
        load_llm_config()["limits"].get("request_deadline_seconds", 15)
    )
    limiter = request.app.state.question_rate_limiter
    if not limiter.allow(
        identity["tenant_id"],
        identity["user_id"],
    ):
        log_answer_audit(
            identity["tenant_id"],
            identity["user_id"],
            [],
            "rate_limited",
            False,
        )
        wait_seconds = limiter.retry_after_seconds(
            identity["tenant_id"], identity["user_id"]
        )
        raise HTTPException(
            status_code=429,
            detail="Too many questions. Please wait before trying again.",
            headers={"Retry-After": str(wait_seconds)},
        )
    block_reason = query_block_reason(payload.question)
    if block_reason is not None:
        log_answer_audit(
            identity["tenant_id"],
            identity["user_id"],
            [],
            f"blocked_{block_reason}",
            False,
        )
        raise HTTPException(
            status_code=400, detail="Question was blocked by safety checks."
        )
    ready_document_ids = status_store.ready_document_ids(
        identity["tenant_id"],
        identity["user_id"],
        identity["role"],
    )

    if not ready_document_ids:
        return {"answer": UNSUPPORTED_ANSWER, "citations": []}

    def check_global_question_cap() -> str | None:
        global_tenant = "global"
        global_user = "questions"
        if not request.app.state.global_question_daily_limiter.allow(
            global_tenant, global_user
        ):
            return "global_daily_cap"
        if not request.app.state.global_question_minute_limiter.allow(
            global_tenant, global_user
        ):
            return "global_minute_cap"
        return None

    global_cap_reason = check_global_question_cap()
    if global_cap_reason is None:
        google_api_key = os.getenv("GOOGLE_API_KEY")
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not google_api_key or not groq_api_key:
            raise HTTPException(
                status_code=503,
                detail="Question answering is not configured.",
            )

        if not hasattr(request.app.state, "answer_providers"):
            request.app.state.answer_providers = create_answer_providers(
                google_api_key,
                groq_api_key,
            )
        primary_provider, backup_provider = request.app.state.answer_providers
    else:
        primary_provider = backup_provider = None

    def generation_cap_check() -> str | None:
        return global_cap_reason

    result = answer_question(
        qdrant_client=request.app.state.qdrant_client,
        primary_provider=primary_provider,
        backup_provider=backup_provider,
        question=payload.question,
        tenant_id=identity["tenant_id"],
        user_id=identity["user_id"],
        role=identity["role"],
        ready_document_ids=ready_document_ids,
        answer_cache=request.app.state.answer_cache,
        document_set_version=status_store.get_document_set_version(
            identity["tenant_id"],
            identity["user_id"],
        ),
        request_deadline=request_deadline,
        generation_cap_check=generation_cap_check,
    )
    record_degraded_response(
        request.app,
        getattr(result, "degraded_reason", None),
    )
    return {
        "answer": result.answer,
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "document_name": citation.document_name,
                "location": citation.location,
                "text": getattr(citation, "text", None),
            }
            for citation in result.citations
        ],
        "degraded": getattr(result, "degraded_mode", False),
        "trace": getattr(result, "trace", None),
    }


def get_document_metrics_summary(
    status_store: DocumentStatusStore, tenant_id: str | None = None
) -> dict[str, int]:
    """Calculate live document counts grouped by status from SQLite."""
    import sqlite3
    from contextlib import closing

    query = "SELECT status, COUNT(*) FROM document_status"
    params = ()
    if tenant_id:
        query += " WHERE tenant_id = ?"
        params = (tenant_id,)
    query += " GROUP BY status"

    counts = {"ready": 0, "failed": 0, "processing": 0, "queued": 0, "total": 0}
    with closing(sqlite3.connect(status_store.database_path)) as connection:
        for row in connection.execute(query, params):
            status, count = row[0], row[1]
            if status in counts:
                counts[status] = count
            counts["total"] += count
    return counts


@app.get("/dashboard/summary")
def get_dashboard_summary(
    request: Request,
    identity: dict[str, str] = Depends(get_current_identity),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict:
    daily_limiter = request.app.state.global_question_daily_limiter
    daily_used = daily_limiter.windows.get(("global", "questions"), (0, 0))[1]
    daily_max = daily_limiter.max_requests
    daily_remaining = max(0, daily_max - daily_used)

    if identity["role"] != "admin":
        my_docs = status_store.list_accessible(
            identity["tenant_id"], identity["user_id"], identity["role"]
        )
        ready_count = sum(1 for d in my_docs if d.get("status") == "ready")
        failed_count = sum(1 for d in my_docs if d.get("status") == "failed")
        return {
            "role": "member",
            "user_id": identity["user_id"],
            "tenant_id": identity["tenant_id"],
            "personal_documents": {
                "total": len(my_docs),
                "ready": ready_count,
                "failed": failed_count,
            },
            "rate_limits": {
                "user_requests_per_minute": 2,
                "daily_pool_remaining": daily_remaining,
                "daily_pool_max": daily_max,
            },
        }

    tenant_docs = get_document_metrics_summary(
        status_store, tenant_id=identity["tenant_id"]
    )
    system_docs = get_document_metrics_summary(status_store)
    cache_stats = request.app.state.answer_cache.stats()
    degraded_counts = dict(request.app.state.degraded_response_counts)

    try:
        request.app.state.qdrant_client.get_collections()
        qdrant_status = "connected"
    except Exception:
        qdrant_status = "unavailable"

    model_status = "loaded" if is_embedding_model_loaded() else "not_loaded"

    return {
        "role": "admin",
        "admin_tenant": identity["tenant_id"],
        "admin_user": identity["user_id"],
        "documents": {
            "tenant": tenant_docs,
            "system_total": system_docs,
        },
        "requests": {
            "daily_questions_used": daily_used,
            "daily_questions_max": daily_max,
            "daily_questions_remaining": daily_remaining,
        },
        "cache": {
            "hits": int(cache_stats.get("hits", 0)),
            "misses": int(cache_stats.get("misses", 0)),
            "hit_rate_pct": round(float(cache_stats.get("hit_rate", 0.0)) * 100, 1),
        },
        "reliability": {
            "degraded_responses": degraded_counts,
        },
        "system": {
            "qdrant": qdrant_status,
            "embedding_model": model_status,
            "app_version": get_app_version(),
        },
    }


from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_frontend_spa(full_path: str):
        file_path = FRONTEND_DIST / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIST / "index.html")
