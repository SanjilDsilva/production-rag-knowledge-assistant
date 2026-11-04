from typing import BinaryIO, Protocol
from pathlib import Path
from uuid import uuid4


MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv"}


class UploadStorage(Protocol):
    def save(
        self,
        tenant_id: str,
        document_id: str,
        extension: str,
        content: BinaryIO,
    ) -> str: ...

    def read(self, storage_key: str) -> BinaryIO: ...

    def delete(self, storage_key: str) -> None: ...


class LocalUploadStorage:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def save(
        self,
        tenant_id: str,
        document_id: str,
        extension: str,
        content: BinaryIO,
    ) -> str:
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError("File type is not allowed")

        folder = self.root / tenant_id / document_id
        folder.mkdir(parents=True, exist_ok=True)
        file_path = folder / f"{uuid4().hex}{extension}"

        total_bytes = 0
        try:
            with file_path.open("xb") as saved_file:
                while True:
                    piece = content.read(64 * 1024)
                    if not piece:
                        break

                    total_bytes += len(piece)
                    if total_bytes > MAX_UPLOAD_BYTES:
                        raise ValueError("File is too large")

                    saved_file.write(piece)
        except Exception:
            if file_path.exists():
                file_path.unlink()
            raise

        return file_path.relative_to(self.root).as_posix()

    def _path_for_key(self, storage_key: str) -> Path:
        key_path = Path(storage_key)

        if (
            not key_path.parts
            or key_path.is_absolute()
            or key_path.drive
            or any(part in {".", ".."} for part in key_path.parts)
        ):
            raise ValueError("Invalid storage key")

        file_path = (self.root / key_path).resolve()
        if self.root not in file_path.parents:
            raise ValueError("Storage key points outside the upload folder")

        return file_path

    def read(self, storage_key: str) -> BinaryIO:
        file_path = self._path_for_key(storage_key)
        return file_path.open("rb")

    def delete(self, storage_key: str) -> None:
        file_path = self._path_for_key(storage_key)
        file_path.unlink(missing_ok=True)
