from fastapi.testclient import TestClient

from production_rag.api import app

import production_rag.api as api
import hashlib
from types import SimpleNamespace
from io import BytesIO
from production_rag.status_store import DocumentStatusStore
from production_rag.storage import LocalUploadStorage
import os
import subprocess
import sys
from production_rag.indexing import IndexResult

from production_rag import vector_store
from production_rag.chunking import Chunk

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
        assert api.app.state.qdrant_client.collection_exists(
            collection_name=api.COLLECTION_NAME
        )


def test_hash_upload_file_hashes_and_rewinds():
    content = BytesIO(b"hello")

    actual_hash = api.hash_upload_file(content)

    assert actual_hash == hashlib.sha256(b"hello").hexdigest()
    assert content.read() == b"hello"


def test_upload_ignores_spoofed_fields_and_keeps_path_safe(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)

    scheduled_tasks = []

    def record_scheduled_task(*arguments):
        scheduled_tasks.append(arguments)

    monkeypatch.setattr(api, "process_document_background", record_scheduled_task)
    with TestClient(api.app) as upload_client:
        response = upload_client.post(
            "/documents",
            headers={"X_Dev_User": "user-a"},
            files={"file": ("../../x.txt", b"safe content", "text/plain")},
            data={
                "tenant_id": "fake-tenant",
                "user_id": "fake-user",
                "source": "benchmark",
            },
        )
        assert response.status_code == 202
        result = response.json()

        assert len(scheduled_tasks) == 1
        assert scheduled_tasks[0][3] == "dev-tenant"
        assert scheduled_tasks[0][4] == "dev-user-a"
        assert scheduled_tasks[0][5] == result["document_id"]

        assert result["filename"] == "../../x.txt"
        assert result["status"] == "queued"

        store = api.app.state.status_store
        user_a_rows = store.list_for_user("dev-tenant", "dev-user-a")
        assert len(user_a_rows) == 1
        assert user_a_rows[0]["original_filename"] == "../../x.txt"
        assert store.list_for_user("fake-tenant", "fake-user") == []

        folder = tmp_path / "uploads" / "dev-tenant" / result["document_id"]
        saved_files = list(folder.iterdir())
        assert len(saved_files) == 1
        assert saved_files[0].name != "../../x.txt"


def test_duplicate_upload_reuses_id_only_for_same_user(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)

    scheduled_document_ids = []

    def record_scheduled_task(
        client,
        storage,
        status_store,
        tenant_id,
        user_id,
        document_id,
        original_filename,
        storage_key,
    ):
        scheduled_document_ids.append(document_id)

    monkeypatch.setattr(api, "process_document_background", record_scheduled_task)
    with TestClient(api.app) as upload_client:
        first = upload_client.post(
            "/documents",
            headers={"X_Dev_User": "user-a"},
            files={"file": ("note.txt", b"same content", "text/plain")},
        )
        second = upload_client.post(
            "/documents",
            headers={"X_Dev_User": "user-a"},
            files={"file": ("note.txt", b"same content", "text/plain")},
        )
        user_b_upload = upload_client.post(
            "/documents",
            headers={"X_Dev_User": "user-b"},
            files={"file": ("note.txt", b"same content", "text/plain")},
        )

        assert scheduled_document_ids == [
            first.json()["document_id"],
            user_b_upload.json()["document_id"],
        ]
        assert first.status_code == 202
        assert second.json()["document_id"] == first.json()["document_id"]
        assert user_b_upload.json()["document_id"] != first.json()["document_id"]

        store = api.app.state.status_store
        assert len(store.list_for_user("dev-tenant", "dev-user-a")) == 1
        assert len(store.list_for_user("dev-tenant", "dev-user-b")) == 1

        folder = tmp_path / "uploads" / "dev-tenant" / first.json()["document_id"]
        assert len(list(folder.iterdir())) == 1


def test_background_indexing_failure_marks_row_failed(tmp_path, monkeypatch):
    storage = LocalUploadStorage(tmp_path / "uploads")
    store = DocumentStatusStore(tmp_path / "app.db")
    storage_key = storage.save(
        "tenant-a",
        "doc-1",
        ".txt",
        BytesIO(b"test file"),
    )
    store.create_queued(
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        "a" * 64,
        storage_key,
    )

    cleanup_calls = []

    def fail_indexing(
        client, path, tenant_id, user_id, document_id, original_filename=None
    ):
        raise ValueError("example error")

    def record_cleanup(client, tenant_id, user_id, document_id):
        cleanup_calls.append((tenant_id, user_id, document_id))

    monkeypatch.setattr(api, "index_document_file", fail_indexing)
    monkeypatch.setattr(api, "delete_document_chunks", record_cleanup)

    api.process_document_background(
        object(),
        storage,
        store,
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        storage_key,
    )

    row = store.list_for_user("tenant-a", "user-a")[0]
    assert row["status"] == "failed"
    assert row["error_message"] == "Indexing failed (ValueError)"
    assert cleanup_calls == [("tenant-a", "user-a", "doc-1")]


def test_background_worker_marks_ready_before_old_chunk_cleanup(tmp_path, monkeypatch):
    storage = LocalUploadStorage(tmp_path / "uploads")
    store = DocumentStatusStore(tmp_path / "app.db")
    storage_key = storage.save("tenant-a", "doc-1", ".txt", BytesIO(b"new version"))
    store.create_queued(
        "tenant-a", "user-a", "doc-1", "report.txt", "a" * 64, storage_key
    )

    index_calls = []
    cleanup_calls = []

    def fake_indexing(
        client, path, tenant_id, user_id, document_id, original_filename=None
    ):
        index_calls.append((path.suffix, original_filename, document_id))
        return IndexResult(document_id, "indexed", "old-doc")

    def record_cleanup(client, tenant_id, user_id, document_id):
        row = store.list_for_user(tenant_id, user_id)[0]
        cleanup_calls.append((document_id, row["status"]))

    monkeypatch.setattr(api, "index_document_file", fake_indexing)
    monkeypatch.setattr(api, "delete_document_chunks", record_cleanup)

    api.process_document_background(
        object(),
        storage,
        store,
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        storage_key,
    )

    assert index_calls == [(".txt", "report.txt", "doc-1")]
    assert cleanup_calls == [("old-doc", "ready")]
    assert store.list_for_user("tenant-a", "user-a")[0]["status"] == "ready"


def test_list_documents_only_returns_current_users_documents(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)

    with TestClient(api.app) as test_client:
        store = api.app.state.status_store
        store.create_queued(
            "dev-tenant", "dev-user-a", "doc-a", "a.txt", "hash-a", "a.txt"
        )
        store.create_queued(
            "dev-tenant", "dev-user-b", "doc-b", "b.txt", "hash-b", "b.txt"
        )

        user_a_response = test_client.get(
            "/documents",
            headers={"X_Dev_User": "user-a"},
        )
        user_b_response = test_client.get(
            "/documents",
            headers={"X_Dev_User": "user-b"},
        )

    assert user_a_response.status_code == 200
    assert user_b_response.status_code == 200
    assert [item["document_id"] for item in user_a_response.json()["documents"]] == [
        "doc-a"
    ]
    assert [item["document_id"] for item in user_b_response.json()["documents"]] == [
        "doc-b"
    ]


def test_delete_document_removes_file_chunks_and_status(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)

    with TestClient(api.app) as test_client:
        storage = api.app.state.upload_storage
        store = api.app.state.status_store
        client = api.app.state.qdrant_client
        file_hash = "f" * 64

        storage_key = storage.save(
            "dev-tenant", "doc-1", ".txt", BytesIO(b"example content")
        )
        store.create_queued(
            "dev-tenant",
            "dev-user-a",
            "doc-1",
            "report.txt",
            file_hash,
            storage_key,
        )

        chunk = Chunk(
            chunk_id="a" * 64,
            document_id="doc-1",
            position=0,
            text="Example content for the test.",
            chunker_version="test-v1",
            tenant_id="dev-tenant",
            user_id="dev-user-a",
            location="page 1",
            content_hash="c" * 64,
        )
        vector_store.upsert_chunks(
            client,
            [chunk],
            [[1.0] + [0.0] * 383],
            filename="report.txt",
            file_hash=file_hash,
        )

        response = test_client.delete(
            "/documents/doc-1",
            headers={"X_Dev_User": "user-a"},
        )

        assert response.status_code == 200
        assert store.get_for_user("dev-tenant", "dev-user-a", "doc-1") is None
        assert list((storage.root / "dev-tenant" / "doc-1").iterdir()) == []
        assert (
            vector_store.find_document_by_file_hash(
                client, "dev-tenant", "dev-user-a", file_hash
            )
            is None
        )


def test_delete_document_cannot_remove_another_tenants_document(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)
    cleanup_calls = []
    monkeypatch.setattr(
        api,
        "delete_document_chunks",
        lambda *arguments: cleanup_calls.append(arguments),
    )

    with TestClient(api.app) as test_client:
        storage = api.app.state.upload_storage
        store = api.app.state.status_store
        storage_key = storage.save(
            "other-tenant", "doc-1", ".txt", BytesIO(b"private content")
        )
        store.create_queued(
            "other-tenant",
            "dev-user-a",
            "doc-1",
            "report.txt",
            "some-hash",
            storage_key,
        )

        response = test_client.delete(
            "/documents/doc-1",
            headers={"X_Dev_User": "user-a"},
        )

        assert response.status_code == 404
        assert store.get_for_user("other-tenant", "dev-user-a", "doc-1") is not None
        assert (storage.root / storage_key).exists()
        assert cleanup_calls == []


def test_question_route_reuses_fake_providers(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(api, "DEV_IDENTITY_ENABLED", True)
    monkeypatch.setenv("GOOGLE_API_KEY", "fake-google-key")
    monkeypatch.setenv("GROQ_API_KEY", "fake-groq-key")
    monkeypatch.delattr(api.app.state, "answer_providers", raising=False)

    primary = object()
    backup = object()
    factory_calls = []
    answer_calls = []

    def fake_create_providers(google_key, groq_key):
        factory_calls.append((google_key, groq_key))
        return primary, backup

    def fake_answer_question(**arguments):
        answer_calls.append(arguments)
        return SimpleNamespace(answer="Fake answer", citations=[])

    monkeypatch.setattr(api, "create_answer_providers", fake_create_providers)
    monkeypatch.setattr(api, "answer_question", fake_answer_question)

    with TestClient(api.app) as test_client:
        store = api.app.state.status_store
        store.create_queued(
            "dev-tenant",
            "dev-user-a",
            "doc-1",
            "sample.txt",
            "sample-hash",
            "dev-tenant/doc-1/file.txt",
        )
        store.update_status("dev-tenant", "dev-user-a", "doc-1", "ready")

        first = test_client.post(
            "/questions",
            headers={"X_Dev_User": "user-a"},
            json={"question": "First question"},
        )
        second = test_client.post(
            "/questions",
            headers={"X_Dev_User": "user-a"},
            json={"question": "Second question"},
        )

    assert first.json()["answer"] == "Fake answer"
    assert second.json()["answer"] == "Fake answer"
    assert factory_calls == [("fake-google-key", "fake-groq-key")]
    assert len(answer_calls) == 2
    assert answer_calls[0]["primary_provider"] is primary
    assert answer_calls[1]["primary_provider"] is primary
    assert answer_calls[0]["backup_provider"] is backup
    assert answer_calls[1]["backup_provider"] is backup


def test_openapi_displays_privacy_notice():
    description = api.app.openapi()["info"]["description"].casefold()

    assert "may use submitted text" in description
    assert "only use public or sample documents" in description
