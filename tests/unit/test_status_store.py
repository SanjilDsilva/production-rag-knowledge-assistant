import sqlite3

from production_rag.status_store import DocumentStatusStore


def test_initialize_creates_table_and_enables_wal(tmp_path):
    database_path = tmp_path / "app.db"
    DocumentStatusStore(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        journal_mode = connection.execute("PRAGMA journal_mode").fetchone()[0]

    assert ("document_status",) in tables
    assert journal_mode.casefold() == "wal"


def test_create_queued_saves_document_owner_and_status(tmp_path):
    database_path = tmp_path / "app.db"
    store = DocumentStatusStore(database_path)

    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="abc123",
        storage_key="tenant-a/doc-1/random.pdf",
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT tenant_id, user_id, document_id, status
            FROM document_status
            """
        ).fetchone()

    assert row == ("tenant-a", "user-a", "doc-1", "queued")


def test_user_cannot_list_another_users_documents(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="abc123",
        storage_key="tenant-a/doc-1/random.pdf",
    )

    assert len(store.list_for_user("tenant-a", "user-a")) == 1
    assert store.list_for_user("tenant-a", "user-b") == []
    assert store.list_for_user("tenant-b", "user-a") == []


def test_user_cannot_update_another_users_document(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="abc123",
        storage_key="tenant-a/doc-1/random.pdf",
    )

    changed = store.update_status("tenant-a", "user-b", "doc-1", "ready")

    assert changed is False
    assert store.list_for_user("tenant-a", "user-a")[0]["status"] == "queued"


def test_reset_processing_puts_stuck_document_back_in_queue(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="abc123",
        storage_key="tenant-a/doc-1/random.pdf",
    )
    store.update_status("tenant-a", "user-a", "doc-1", "processing")

    changed_count = store.reset_processing()

    assert changed_count == 1
    assert store.list_for_user("tenant-a", "user-a")[0]["status"] == "queued"


def test_user_cannot_delete_another_users_status_row(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="abc123",
        storage_key="tenant-a/doc-1/random.pdf",
    )

    deleted = store.delete_for_user("tenant-a", "user-b", "doc-1")

    assert deleted is False
    assert len(store.list_for_user("tenant-a", "user-a")) == 1


def test_file_hash_lookup_is_scoped_to_tenant_and_user(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        tenant_id="tenant-a",
        user_id="user-a",
        document_id="doc-1",
        original_filename="report.pdf",
        file_hash="same-hash",
        storage_key="tenant-a/doc-1/random.pdf",
    )

    found = store.find_active_by_file_hash("tenant-a", "user-a", "same-hash")

    assert found is not None
    assert found["document_id"] == "doc-1"
    assert store.find_active_by_file_hash("tenant-a", "user-b", "same-hash") is None
    assert store.find_active_by_file_hash("tenant-b", "user-a", "same-hash") is None


def test_get_for_user_requires_matching_tenant_and_user(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        "file-hash",
        "tenant-a/doc-1/random.txt",
    )

    owner = store.get_for_user("tenant-a", "user-a", "doc-1")
    wrong_user = store.get_for_user("tenant-a", "user-b", "doc-1")
    wrong_tenant = store.get_for_user("tenant-b", "user-a", "doc-1")

    assert owner is not None
    assert owner["storage_key"] == "tenant-a/doc-1/random.txt"
    assert wrong_user is None
    assert wrong_tenant is None


def test_ready_document_ids_include_only_ready_documents_for_that_user(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        "tenant-a", "user-a", "doc-ready", "ready.txt", "hash-1", "ready.txt"
    )
    store.create_queued(
        "tenant-a", "user-a", "doc-queued", "queued.txt", "hash-2", "queued.txt"
    )
    store.create_queued(
        "tenant-a", "user-b", "other-user-doc", "other.txt", "hash-3", "other.txt"
    )

    store.update_status("tenant-a", "user-a", "doc-ready", "ready")
    store.update_status("tenant-a", "user-b", "other-user-doc", "ready")

    assert store.ready_document_ids("tenant-a", "user-a") == ["doc-ready"]


def test_new_upload_bumps_only_its_owners_document_set_version(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")

    store.create_queued(
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        "hash-1",
        "tenant-a/doc-1/file.txt",
    )

    assert store.get_document_set_version("tenant-a", "user-a") == "1"
    assert store.get_document_set_version("tenant-a", "user-b") == "0"
    assert store.get_document_set_version("tenant-b", "user-a") == "0"


def test_ready_status_bumps_document_set_version(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        "hash-1",
        "tenant-a/doc-1/file.txt",
    )

    changed = store.update_status(
        "tenant-a",
        "user-a",
        "doc-1",
        "ready",
    )

    assert changed is True
    assert store.get_document_set_version("tenant-a", "user-a") == "2"


def test_delete_bumps_only_the_owner_document_set_version(tmp_path):
    store = DocumentStatusStore(tmp_path / "app.db")
    store.create_queued(
        "tenant-a",
        "user-a",
        "doc-1",
        "report.txt",
        "hash-1",
        "tenant-a/doc-1/file.txt",
    )

    deleted = store.delete_for_user("tenant-a", "user-a", "doc-1")

    assert deleted is True
    assert store.get_document_set_version("tenant-a", "user-a") == "2"
    assert store.get_document_set_version("tenant-a", "user-b") == "0"
    assert store.get_document_set_version("tenant-b", "user-a") == "0"
