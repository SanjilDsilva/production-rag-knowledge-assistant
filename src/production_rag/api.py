import os
from pathlib import Path
from contextlib import asynccontextmanager
import hashlib
from typing import BinaryIO
from qdrant_client import QdrantClient
import shutil
from tempfile import TemporaryDirectory

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
)
from uuid import uuid4

from production_rag.providers import create_answer_providers
from pydantic import BaseModel, Field
from production_rag.answering import UNSUPPORTED_ANSWER, answer_question
from production_rag.answer_cache import BoundedAnswerCache

class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


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

    if x_dev_user not in ("user-a", "user-b"):
        raise HTTPException(
            status_code=400,
            detail="X_Dev_User must be user-a or user-b.",
        )

    return {
        "tenant_id": "dev-tenant",
        "user_id": f"dev-{x_dev_user}",
    }


DATA_ROOT = Path(os.getenv("RAG_DATA_DIR", "data"))


def initialize_app_storage(app: FastAPI) -> None:
    app.state.upload_storage = LocalUploadStorage(DATA_ROOT / "uploads")
    app.state.status_store = DocumentStatusStore(DATA_ROOT / "app.db")
    app.state.answer_cache = BoundedAnswerCache()
    app.state.status_store.reset_processing()
    qdrant_client = QdrantClient(path=str(DATA_ROOT / "qdrant"))
    create_collection(qdrant_client)
    app.state.qdrant_client = qdrant_client


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
    identity: dict[str, str] = Depends(get_current_identity),
    storage: UploadStorage = Depends(get_upload_storage),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, str]:
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


@app.get("/documents")
def list_documents(
    identity: dict[str, str] = Depends(get_current_identity),
    status_store: DocumentStatusStore = Depends(get_document_status_store),
) -> dict[str, list[dict[str, str | None]]]:
    documents = status_store.list_for_user(
        identity["tenant_id"],
        identity["user_id"],
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
    document = status_store.get_for_user(
        identity["tenant_id"],
        identity["user_id"],
        document_id,
    )
    if document is None:
        raise HTTPException(status_code=404, detail="Document not found")

    delete_document_chunks(
        request.app.state.qdrant_client,
        identity["tenant_id"],
        identity["user_id"],
        document_id,
    )

    storage.delete(document["storage_key"])
    status_store.delete_for_user(
        identity["tenant_id"],
        identity["user_id"],
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
    ready_document_ids = status_store.ready_document_ids(
        identity["tenant_id"],
        identity["user_id"],
    )

    if not ready_document_ids:
        return {"answer": UNSUPPORTED_ANSWER, "citations": []}

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
    result = answer_question(
        qdrant_client=request.app.state.qdrant_client,
        primary_provider=primary_provider,
        backup_provider=backup_provider,
        question=payload.question,
        tenant_id=identity["tenant_id"],
        user_id=identity["user_id"],
        ready_document_ids=ready_document_ids,
        answer_cache=request.app.state.answer_cache,
        document_set_version=status_store.get_document_set_version(
            identity["tenant_id"],
            identity["user_id"],
        ),
    )
    return {
        "answer": result.answer,
        "citations": [
            {
                "chunk_id": citation.chunk_id,
                "document_name": citation.document_name,
                "location": citation.location,
            }
            for citation in result.citations
        ],
    }
