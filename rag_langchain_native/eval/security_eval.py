"""不调用 Gemini 的 Phase 9 权限/租户确定性评估。"""

from __future__ import annotations

import tempfile
from pathlib import Path

from ..catalog import DocumentCatalog, NotFoundError
from ..security import AuthService, AuthenticationError, build_authorization_scope
from .common import RESULTS_DIR, write_json


RESULT_FILE = RESULTS_DIR / "phase9_security_evaluation.json"


def _publish(catalog, tenant, source, classification, owner):
    catalog.ensure_knowledge_base(tenant, "default")
    document, _ = catalog.register_document(
        tenant_id=tenant, knowledge_base_id="default", document_name=source,
        source=source, document_type="pdf", classification=classification, owner=owner,
    )
    version, _ = catalog.create_version(
        document_id=document["document_id"], content_hash=f"hash-{tenant}-{source}",
        storage_path=f"/{tenant}/{source}",
    )
    catalog.set_version_status(version["version_id"], "indexed")
    catalog.publish_version(document["document_id"], version["version_id"])
    return document


def run() -> dict:
    checks = []
    with tempfile.TemporaryDirectory() as directory:
        catalog = DocumentCatalog(Path(directory) / "security.sqlite3")
        auth = AuthService(catalog, "security-evaluation-secret", 3600)
        users = {}
        for tenant, name, roles in (
            ("tenant-a", "general", ["general"]),
            ("tenant-a", "hr", ["hr"]),
            ("tenant-a", "admin", ["admin"]),
            ("tenant-b", "admin-b", ["admin"]),
        ):
            auth.register_user(
                tenant_id=tenant, username=name, password="password123", roles=roles
            )
            users[name] = auth.verify_token(auth.login(tenant, name, "password123"))
        general = _publish(catalog, "tenant-a", "general.pdf", "general", users["admin"].user_id)
        hr = _publish(catalog, "tenant-a", "hr.pdf", "hr", users["admin"].user_id)
        other = _publish(catalog, "tenant-b", "same.pdf", "general", users["admin-b"].user_id)

        general_scope = build_authorization_scope(catalog, users["general"], "default")
        hr_scope = build_authorization_scope(catalog, users["hr"], "default")
        other_scope = build_authorization_scope(catalog, users["admin-b"], "default")
        checks.extend(
            [
                {"name": "general_can_read_general", "passed": general["document_id"] in general_scope.document_ids},
                {"name": "general_cannot_read_hr", "passed": hr["document_id"] not in general_scope.document_ids},
                {"name": "hr_can_read_hr", "passed": hr["document_id"] in hr_scope.document_ids},
                {"name": "tenant_a_cannot_read_tenant_b", "passed": other["document_id"] not in general_scope.document_ids},
                {"name": "tenant_b_only_reads_tenant_b", "passed": set(other_scope.document_ids) == {other["document_id"]}},
            ]
        )
        token = auth.login("tenant-a", "general", "password123")
        try:
            auth.verify_token(token + "tampered")
            tamper_blocked = False
        except AuthenticationError:
            tamper_blocked = True
        checks.append({"name": "tampered_token_rejected", "passed": tamper_blocked})
        conversation = catalog.create_conversation(
            "tenant-a", users["general"].user_id, "default"
        )
        try:
            catalog.get_conversation(
                conversation["conversation_id"],
                "tenant-a",
                users["hr"].user_id,
            )
            ownership_blocked = False
        except NotFoundError:
            ownership_blocked = True
        checks.append({"name": "conversation_ownership_enforced", "passed": ownership_blocked})

    output = {
        "summary": {
            "total": len(checks),
            "passed": sum(item["passed"] for item in checks),
            "failed": sum(not item["passed"] for item in checks),
        },
        "checks": checks,
        "note": "This deterministic evaluation complements, but does not replace, penetration testing.",
    }
    write_json(RESULT_FILE, output)
    return output


if __name__ == "__main__":
    print(run()["summary"])

