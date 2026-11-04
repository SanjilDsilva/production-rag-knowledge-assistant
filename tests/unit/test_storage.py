import io

import pytest

import production_rag.storage as storage


def test_save_writes_file_with_generated_name(tmp_path):
    file_storage = storage.LocalUploadStorage(tmp_path)

    storage_key = file_storage.save(
        "tenant-a",
        "document-1",
        ".txt",
        io.BytesIO(b"hello"),
    )

    saved_path = tmp_path.joinpath(*storage_key.split("/"))

    assert storage_key.startswith("tenant-a/document-1/")
    assert len(saved_path.stem) == 32
    assert saved_path.read_bytes() == b"hello"


def test_save_removes_partial_file_when_size_limit_is_exceeded(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "MAX_UPLOAD_BYTES", 4)
    file_storage = storage.LocalUploadStorage(tmp_path)

    with pytest.raises(ValueError, match="File is too large"):
        file_storage.save(
            "tenant-a",
            "document-1",
            ".txt",
            io.BytesIO(b"12345"),
        )

    folder = tmp_path / "tenant-a" / "document-1"
    assert list(folder.iterdir()) == []


def test_read_and_delete_saved_file(tmp_path):
    file_storage = storage.LocalUploadStorage(tmp_path)
    storage_key = file_storage.save(
        "tenant-a",
        "document-1",
        ".txt",
        io.BytesIO(b"hello"),
    )

    with file_storage.read(storage_key) as saved_file:
        assert saved_file.read() == b"hello"

    file_storage.delete(storage_key)
    saved_path = tmp_path.joinpath(*storage_key.split("/"))
    assert not saved_path.exists()


def test_read_rejects_path_outside_upload_folder(tmp_path):
    file_storage = storage.LocalUploadStorage(tmp_path)

    with pytest.raises(ValueError, match="Invalid storage key"):
        file_storage.read("../../outside.txt")
