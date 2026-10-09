"""LangChain V3 的命令行入口。

文件职责：
    保留 Phase 8 的 ``ingest/retrieve/ask/health``，并提供 Phase 9 的本地用户创建、
    企业上传、授权检索和授权问答子命令。

在 RAG Pipeline 中的位置：
    CLI 与 FastAPI 都是最上游入口。它只解析命令行参数并调用正式业务函数，不重新实现
    Ingestion 或 RAG。因此同一问题从 CLI 与 API 进入时使用同一个 V3 Service。

输入与输出：
    输入来自命令行字符串；输出统一用 UTF-8 JSON 打印到终端。

调用关系：
    ``ingest`` -> ``ingest_pdf/rebuild_knowledge_base``；
    ``retrieve`` -> ``NativeRAGService.retrieve_only``；
    ``ask`` -> ``NativeRAGService.ask_rag``；
    ``health`` 只返回配置，不加载 Gemini；Enterprise 命令通过 Catalog 验证用户后调用
    ``DocumentLifecycleService`` 或 ``EnterpriseRAGService``。

LangChain 关系：
    argparse/CLI 不是 LangChain；``ask`` 最终进入 LCEL 的 ``chain.invoke``。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .chain import get_service
from .catalog import DocumentCatalog
from .config import DEFAULT_SETTINGS
from .enterprise import EnterpriseRAGService
from .ingestion import ingest_pdf, rebuild_knowledge_base
from .lifecycle import DocumentLifecycleService
from .security import AuthService
from .backup import BackupService


def build_parser() -> argparse.ArgumentParser:
    """声明 V3 CLI 的子命令和参数。

    返回：
        argparse.ArgumentParser: 可把 ``list[str]`` 命令行解析为带属性的 Namespace。

    ``required=True`` 要求用户必须选择子命令；``store_true`` 表示出现 ``--rebuild`` 时
    值为 True；``nargs="?"`` 表示 PDF 路径可以省略。
    """
    parser = argparse.ArgumentParser(description="LangChain Native RAG V3")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Ingest PDFs into V3 Chroma")
    ingest.add_argument("path", nargs="?", help="PDF file; omit to rebuild data/pdf")
    ingest.add_argument(
        "--rebuild", action="store_true", help="Rebuild the V3 collection"
    )

    retrieve = commands.add_parser("retrieve", help="Retrieval-only query")
    retrieve.add_argument("question")
    retrieve.add_argument("--top-k", type=int, default=10)

    ask = commands.add_parser("ask", help="Run the complete RAG chain")
    ask.add_argument("question")

    commands.add_parser("health", help="Show V3 runtime configuration")
    backup = commands.add_parser("backup", help="Create a consistent V3 runtime backup")
    backup.add_argument("destination")
    verify = commands.add_parser("verify-backup", help="Verify backup hashes")
    verify.add_argument("backup_dir")
    restore = commands.add_parser("restore-backup", help="Restore into a new empty directory")
    restore.add_argument("backup_dir")
    restore.add_argument("destination")

    create_user = commands.add_parser(
        "create-user", help="Create a Phase 9 local development user"
    )
    create_user.add_argument("--tenant", required=True)
    create_user.add_argument("--username", required=True)
    create_user.add_argument("--password", required=True)
    create_user.add_argument(
        "--roles", default="general", help="Comma-separated roles"
    )

    enterprise_upload = commands.add_parser(
        "enterprise-upload", help="Upload, index and optionally publish a PDF"
    )
    enterprise_upload.add_argument("path")
    enterprise_upload.add_argument("--tenant", required=True)
    enterprise_upload.add_argument("--username", required=True)
    enterprise_upload.add_argument("--password", required=True)
    enterprise_upload.add_argument("--knowledge-base", default="default")
    enterprise_upload.add_argument("--classification", default="general")
    enterprise_upload.add_argument("--publish", action="store_true")

    for name in ("enterprise-retrieve", "enterprise-ask"):
        command = commands.add_parser(name, help=f"Run Phase 9 {name}")
        command.add_argument("question")
        command.add_argument("--tenant", required=True)
        command.add_argument("--username", required=True)
        command.add_argument("--password", required=True)
        command.add_argument("--knowledge-base", default="default")
        if name == "enterprise-retrieve":
            command.add_argument("--top-k", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> int:
    """解析命令行，调用对应 V3 正式入口并打印 JSON。

    参数：
        argv (list[str] | None): 测试可传自定义参数列表；None 时 argparse 读取真实命令行。

    返回：
        int: 进程退出码，0 表示命令成功完成。

    执行过程：
        1. Windows 终端支持时把 stdout 调整为 UTF-8。
        2. 解析子命令。
        3. Ingest 后清除 ``get_service`` 缓存，避免继续使用旧 BM25/Chain。
        4. Retrieve 不调用 Gemini；Ask 执行完整 RAG；Health 只读配置。
        5. Enterprise 命令先使用密码取得并验证本地签名 Token 对应的 Principal。
        6. ``json.dumps(..., ensure_ascii=False)`` 以可读中文打印结果。

    初学者知识点：
        ``if/elif/else`` 保证一次只运行一个子命令。函数末尾返回整数；模块入口使用
        ``raise SystemExit(main())`` 把它变成操作系统可见的退出码。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        if args.rebuild or not args.path:
            result = rebuild_knowledge_base()
        else:
            result = ingest_pdf(args.path)
        get_service.cache_clear()
    elif args.command == "retrieve":
        result = get_service().retrieve_only(args.question, top_k=args.top_k)
    elif args.command == "ask":
        result = get_service().ask_rag(args.question)
    elif args.command == "backup":
        result = {"backup_dir": str(BackupService().create(args.destination))}
    elif args.command == "verify-backup":
        result = BackupService.verify(args.backup_dir)
    elif args.command == "restore-backup":
        result = {"restore_dir": str(BackupService().restore(args.backup_dir, args.destination))}
    elif args.command == "create-user":
        catalog = DocumentCatalog(DEFAULT_SETTINGS.catalog_path)
        auth = AuthService(
            catalog,
            DEFAULT_SETTINGS.auth_secret,
            DEFAULT_SETTINGS.auth_token_ttl_seconds,
        )
        result = auth.register_user(
            tenant_id=args.tenant,
            username=args.username,
            password=args.password,
            roles=[role.strip() for role in args.roles.split(",") if role.strip()],
        )
        result.pop("password_hash", None)
    elif args.command in {
        "enterprise-upload",
        "enterprise-retrieve",
        "enterprise-ask",
    }:
        catalog = DocumentCatalog(DEFAULT_SETTINGS.catalog_path)
        auth = AuthService(
            catalog,
            DEFAULT_SETTINGS.auth_secret,
            DEFAULT_SETTINGS.auth_token_ttl_seconds,
        )
        principal = auth.verify_token(
            auth.login(args.tenant, args.username, args.password)
        )
        if args.command == "enterprise-upload":
            if not principal.is_admin:
                raise PermissionError("Administrator role required")
            lifecycle = DocumentLifecycleService(catalog, DEFAULT_SETTINGS)
            path = Path(args.path)
            document = lifecycle.register_document(
                tenant_id=principal.tenant_id,
                knowledge_base_id=args.knowledge_base,
                document_name=path.name,
                source=path.name,
                classification=args.classification,
                owner=principal.user_id,
            )
            version = lifecycle.upload_version(
                tenant_id=principal.tenant_id,
                knowledge_base_id=args.knowledge_base,
                document_id=document["document_id"],
                filename=path.name,
                content=path.read_bytes(),
            )
            if version["status"] in {"registered", "failed"}:
                version = lifecycle.index_version(
                    document["document_id"], version["version_id"]
                )
            if args.publish:
                lifecycle.publish_version(
                    document["document_id"], version["version_id"]
                )
            result = {"document": document, "version": version}
        else:
            service = EnterpriseRAGService(catalog, DEFAULT_SETTINGS)
            values = {
                "principal": principal,
                "knowledge_base_id": args.knowledge_base,
                "question": args.question,
            }
            result = (
                service.retrieve_only(**values, top_k=args.top_k)
                if args.command == "enterprise-retrieve"
                else service.chat(**values)
            )
    else:
        result = {
            "status": "ok",
            "version": "v3",
            "runtime": str(DEFAULT_SETTINGS.runtime_dir),
            "collection": DEFAULT_SETTINGS.collection_name,
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    # 只有 ``python -m rag_langchain_native.cli`` 直接运行本模块时才进入这里；import 不会执行。
    raise SystemExit(main())

