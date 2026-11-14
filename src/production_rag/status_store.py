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
                    document_id TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    storage_key TEXT NOT NULL,
                    status TEXT NOT NULL
                        CHECK (status IN ('queued', 'processing', 'ready', 'failed')),
                    error_message TEXT,
                    visibility TEXT NOT NULL DEFAULT 'private'
                    CHECK (visibility IN ('private', 'tenant_shared')),
                    PRIMARY KEY (tenant_id, user_id, document_id)
                    )
                    """
            )

            columns = {
                row[1]
                for row in connection.execute("PRAGMA table_info(document_status)")
            }
            if "visibility" not in columns:
                connection.execute(
                    """
                    ALTER TABLE document_status
                    ADD COLUMN visibility TEXT NOT NULL DEFAULT 'private'
                    CHECK (visibility IN ('private', 'tenant_shared'))
                    """
                )

            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS document_set_versions (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                version INTEGER NOT NULL,
                PRIMARY KEY (tenant_id, user_id)
                )
                """
            )
            connection.commit()

    def get_document_set_version(self, tenant_id: str, user_id: str) -> str:
        with closing(sqlite3.connect(self.database_path)) as connection:
            row = connection.execute(
                """
                SELECT version
                FROM document_set_versions
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchone()

        return str(row[0]) if row is not None else "0"

    def _bump_document_set_version(
        self,
        connection: sqlite3.Connection,
        tenant_id: str,
        user_id: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO document_set_versions (tenant_id, user_id, version)
            VALUES (?, ?, 1)
            ON CONFLICT (tenant_id, user_id)
            DO UPDATE SET version = version + 1
            """,
            (tenant_id, user_id),
        )

    def create_queued(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
        original_filename: str,
        file_hash: str,
        storage_key: str,
        visibility: str = "private",
    ) -> None:
        if visibility not in {"private", "tenant_shared"}:
            raise ValueError("Document visibility is not allowed")

        with closing(sqlite3.connect(self.database_path)) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO document_status (
                        tenant_id, user_id, document_id, original_filename,
                        file_hash, storage_key, status, error_message, visibility
                    )
                    VALUES (?, ?, ?, ?, ?, ?, 'queued', NULL, ?)
                    """,
                    (
                        tenant_id,
                        user_id,
                        document_id,
                        original_filename,
                        file_hash,
                        storage_key,
                        visibility,
                    ),
                )
                self._bump_document_set_version(connection, tenant_id, user_id)

    def list_for_user(
        self,
        tenant_id: str,
        user_id: str,
    ) -> list[dict[str, str | None]]:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                """
                SELECT document_id AS document_id, original_filename, status, error_message
                FROM document_status
                WHERE tenant_id = ? AND user_id = ?
                """,
                (tenant_id, user_id),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_accessible(
        self,
        tenant_id: str,
        user_id: str,
        role: str,
    ) -> list[dict[str, str | None]]:
        if role == "admin":
            query = """
                SELECT document_id, original_filename, status, error_message, visibility
                FROM document_status
                WHERE tenant_id = ?
            """
            parameters = (tenant_id,)
        elif role == "member":
            query = """
                SELECT  document_id, original_filename, status, error_message, visibility
                FROM document_status
                WHERE tenant_id = ? AND (user_id = ? OR visibility = 'tenant_shared')    
            """
            parameters = (tenant_id, user_id)
        else:
            return []

        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(query, parameters).fetchall()

        return [dict(row) for row in rows]

    def get_for_user(
        self,
        tenant_id: str,
        user_id: str,
        document_id: str,
    ) -> dict[str, str] | None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT document_id, original_filename, storage_key, status, visibility
                FROM document_status
                WHERE tenant_id = ? AND user_id = ? AND document_id = ?
                """,
                (tenant_id, user_id, document_id),
            ).fetchone()

            return dict(row) if row is not None else None

    def get_visible_document(
        self,
        tenant_id: str,
        user_id: str,
        role: str,
        document_id: str,
    ) -> dict[str, str] | None:
        if role == "admin":
            query = """
                SELECT tenant_id, user_id, document_id, storage_key, visibility
                FROM document_status
                WHERE tenant_id = ? AND document_id = ?
            """
            parameters = (tenant_id, document_id)
        elif role == "member":
            query = """
                SELECT tenant_id, user_id, document_id, storage_key, visibility
                FROM document_status
                WHERE tenant_id = ? AND document_id = ? AND (user_id = ? OR visibility = 'tenant_shared')
            """
            parameters = (tenant_id, document_id, user_id)
        else:
            return None
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(query, parameters).fetchone()

        return dict(row) if row is not None else None

    def ready_document_ids(
        self,
        tenant_id: str,
        user_id: str,
        role: str = "member",
    ) -> list[str]:
        if role == "admin":
            query = """
                SELECT document_id
                FROM document_status
                WHERE tenant_id = ? AND status = 'ready'    
            """
            parameters = (tenant_id,)
        elif role == "member":
            query = """
                SELECT document_id
                FROM document_status
                WHERE tenant_id = ? AND status = 'ready'
                AND (user_id = ? OR visibility = 'tenant_shared')
            """
            parameters = (tenant_id, user_id)
        else:
            return []
        with closing(sqlite3.connect(self.database_path)) as connection:
            rows = connection.execute(query, parameters).fetchall()

        return [row[0] for row in rows]

    def find_active_by_file_hash(
        self,
        tenant_id: str,
        user_id: str,
        file_hash: str,
    ) -> dict[str, str | None] | None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT document_id AS document_id, original_filename, status, error_message
                FROM document_status
                WHERE tenant_id = ? AND user_id = ? AND file_hash = ?
                    AND status IN ('queued', 'processing', 'ready')
                ORDER BY rowid DESC
                LIMIT 1
                """,
                (tenant_id, user_id, file_hash),
            ).fetchone()

            return dict(row) if row is not None else None

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
                if result.rowcount == 1 and status == "ready":
                    self._bump_document_set_version(
                        connection,
                        tenant_id,
                        user_id,
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
                if result.rowcount == 1:
                    self._bump_document_set_version(
                        connection,
                        tenant_id,
                        user_id,
                    )
        return result.rowcount == 1
