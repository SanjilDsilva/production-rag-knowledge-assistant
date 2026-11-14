import pytest

from production_rag.access import can_access_document


@pytest.mark.parametrize(
    ("identity", "document", "expected"),
    [
        (
            {"tenant_id": "tenant-a", "user_id": "user-a", "role": "member"},
            {"tenant_id": "tenant-a", "user_id": "user-a", "visibility": "private"},
            True,
        ),
        (
            {"tenant_id": "tenant-a", "user_id": "user-a", "role": "member"},
            {
                "tenant_id": "tenant-a",
                "user_id": "user-b",
                "visibility": "tenant_shared",
            },
            True,
        ),
        (
            {"tenant_id": "tenant-a", "user_id": "user-a", "role": "member"},
            {"tenant_id": "tenant-a", "user_id": "user-b", "visibility": "private"},
            False,
        ),
        (
            {"tenant_id": "tenant-a", "user_id": "admin-a", "role": "admin"},
            {"tenant_id": "tenant-a", "user_id": "user-b", "visibility": "private"},
            True,
        ),
        (
            {"tenant_id": "tenant-a", "user_id": "admin-a", "role": "admin"},
            {
                "tenant_id": "tenant-b",
                "user_id": "user-b",
                "visibility": "tenant_shared",
            },
            False,
        ),
    ],
)
def test_document_access_rules(identity, document, expected):
    assert can_access_document(identity, document) is expected
