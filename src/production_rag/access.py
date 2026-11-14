from collections.abc import Mapping


def can_access_document(
    identity: Mapping[str, str],
    document: Mapping[str, str],
) -> bool:
    tenant_id = identity.get("tenant_id")
    if not tenant_id or tenant_id != document.get("tenant_id"):
        return False

    role = identity.get("role")
    visibility = document.get("visibility")
    if visibility not in {"private", "tenant_shared"}:
        return False

    if role == "admin":
        return True
    if role != "member":
        return False

    if visibility == "tenant_shared":
        return True
    return identity.get("user_id") == document.get("user_id")
