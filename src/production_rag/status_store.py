import sqlite3
from contextlib import closing
from pathlib import Path


class DocumentStatusStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    def initialize(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                    CREATE TABLE IF NOT EXISTS document_status (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    document_Id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
                    error_message TEXT,
                    PRIMARY KEY (tenant_id, user_id, document_id)
                    )
                    """
            )
            connection.commit()

    def create_queued(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
        original_filename: str,
        file_hash: str,
        storage_key: str,
    ) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO document_status (
                        tenant_id, user_id, document_id, original_filename,
                        file_hash, storage_key, status, error_message
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL)
                    """,
                    (
                        tenant_id,
                        user_id,
                        document_id,
                        original_filename,
                        file_hash,
                        storage_key,
                    ),
                )

    def list_for_user(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[dict[str, str | None]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT document_id, original_filename, status, error_message
                FROM document_status
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchall()

        return [dict(row) for row in rows]

    def update_status(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
        status: str,
        error_message: str | None = None,
    ) -> bool:
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                result = connection.execute(
                    """
                    UPDATE document_status
                    SET status = ?, error_message = ?
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                    """,
                    (
                        status,
                        error_message,
                        tenant_id,
                        user_id,
                        document_id,
                    ),
                )

        return result.rowcount == 1

    def reset_processing(self) -> int:
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                result = connection.execute(
                    """
                    UPDATE document_status
                    SET status = 'queued', error_message = NULL
                    WHERE status = 'processing'
                    """
                )
            return result.rowcount

    def delete_for_user(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
    ) -> bool:
        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                result = connection.execute(
                    """
                    DELETE FROM document_status
                    WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                    """,
                    (tenant_id, user_id, document_id),
                )
        return result.rowcount == 1
