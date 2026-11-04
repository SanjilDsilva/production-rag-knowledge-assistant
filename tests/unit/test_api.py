from fastapi.testclient import TestClient

from production_rag.api import app

import production_rag.api as api

import os
import subprocess
import sys

client = TestClient(app)


def test_health_endpoint_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_dev_identity_returns_server_owned_identity(monkeypatch):
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)

    user_a = api.get_current_identity("user-a")
    user_b = api.get_current_identity("user-b")

    assert user_a == {"tenant_id": "dev-tenant", "user_id": "dev-user-a"}
    assert user_b == {"tenant_id": "dev-tenant", "user_id": "dev-user-b"}


def test_production_refuses_development_identity():
    environment = os.environ.copy()
    environment["APP_ENV"] = "production"
    environment["RAG_DEV_IDENTITY_ENABLED"] = "true"

    result = subprocess.run(
        [sys.executable, "-c", "import production_rag.api"],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Development identity can only run in development." in result.stderr


def test_startup_prepares_local_storage_and_status_store(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)

    with TestClient(api.app):
        assert api.app.state.upload_storage.root == (tmp_path / "uploads").resolve()
        assert api.app.state.status_store.database_path == tmp_path / "app.db"
