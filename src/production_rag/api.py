import os
from pathlib import Path
from contextlib import asynccontextmanager

from production_rag.status_store import DocumentStatusStore
from production_rag.storage import LocalUploadStorage
from fastapi import Depends, FastAPI, Header, HTTPException

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
    app.state.status_store.reset_processing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_app_storage(app)
    yield


app = FastAPI(title="Production RAG", lifespan=lifespan)


@app.get("/health")
def health_endpoint() -> dict[str, str]:
    return {"status": "ok"}
