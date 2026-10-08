from pathlib import Path
import subprocess

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from rag_langchain_native import api
from rag_langchain_native.chain import NativeRAGService
from rag_langchain_native.cli import build_parser


def test_cli_parser_supports_required_commands():
    parser = build_parser()
    assert parser.parse_args(["health"]).command == "health"
    assert parser.parse_args(["retrieve", "问题"]).top_k == 10
    assert parser.parse_args(["ask", "问题"]).question == "问题"
    assert parser.parse_args(["ingest", "--rebuild"]).rebuild is True


def test_fastapi_upload_and_retrieve_share_store(
    monkeypatch, fake_store, v3_settings
):
    model = RunnableLambda(lambda _: AIMessage(content="测试答案"))
    service = NativeRAGService(
        settings=v3_settings,
        vectorstore=fake_store,
        llm=model,
    )
    monkeypatch.setattr(api, "get_service", lambda: service)
    client = TestClient(api.create_app(v3_settings))
    assert client.get("/health").json()["status"] == "ok"

    pdf = v3_settings.project_root / "data" / "pdf" / "travel_policy.pdf"
    with pdf.open("rb") as file:
        response = client.post(
            "/upload", files={"file": (pdf.name, file, "application/pdf")}
        )
    assert response.status_code == 200
    assert fake_store._collection.count() > 0

    response = client.post(
        "/retrieve", json={"question": "国际出差", "top_k": 3}
    )
    assert response.status_code == 200
    assert response.json()["documents"]


def test_original_versions_and_ground_truth_are_unmodified():
    root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            ".",
            ":(exclude)rag_langchain_native/**",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == ""
    import hashlib
    digest = hashlib.sha256((root / "eval" / "test_case.json").read_bytes()).hexdigest()
    assert digest == "e6cff33933ca14f177d2319b91be5e2249b742239a7aa57fa30e53a4054a057a"

