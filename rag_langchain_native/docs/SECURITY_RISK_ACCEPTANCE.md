# Temporary Dependency Risk Acceptance

Last reviewed: 2026-10-09. Review/expiry: 2026-11-09.

`pip-audit` reports `PYSEC-2026-311`, `PYSEC-2026-3813`, `PYSEC-2026-3814` and
`PYSEC-2026-3815` for `chromadb==1.5.9`. At review time PyPI has no newer release
or listed fix version, so a compatible patch cannot be installed.

This is a **temporary, scoped acceptance**, not a claim that the package is safe:

- V3 uses embedded `PersistentClient` through `langchain_chroma`; it does not start or expose Chroma's Python HTTP API.
- Docker publishes only FastAPI on loopback; no Chroma port is published.
- Tenant/version authorization is computed in the application and applied to both Vector and BM25; output scope is checked again.
- Security regression tests enforce cross-tenant and unauthorized retrieval boundaries.

CI still runs `pip-audit`; only these four exact IDs are temporarily allowed so any new advisory fails the Gate. Remove the exceptions as soon as a compatible fixed Chroma release exists. Production deployment should additionally isolate the runtime at the OS/network layer and must never expose Chroma directly.
