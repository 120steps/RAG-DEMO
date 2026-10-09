"""LangChain V3 的命令行入口。

文件职责：
    提供 PowerShell 友好的 ``ingest``、``retrieve``、``ask``、``health`` 子命令。

在 RAG Pipeline 中的位置：
    CLI 与 FastAPI 都是最上游入口。它只解析命令行参数并调用正式业务函数，不重新实现
    Ingestion 或 RAG。因此同一问题从 CLI 与 API 进入时使用同一个 V3 Service。

输入与输出：
    输入来自命令行字符串；输出统一用 UTF-8 JSON 打印到终端。

调用关系：
    ``ingest`` -> ``ingest_pdf/rebuild_knowledge_base``；
    ``retrieve`` -> ``NativeRAGService.retrieve_only``；
    ``ask`` -> ``NativeRAGService.ask_rag``；
    ``health`` 只返回配置，不加载 Gemini。

LangChain 关系：
    argparse/CLI 不是 LangChain；``ask`` 最终进入 LCEL 的 ``chain.invoke``。
"""

from __future__ import annotations

import argparse
import json
import sys

from .chain import get_service
from .config import DEFAULT_SETTINGS
from .ingestion import ingest_pdf, rebuild_knowledge_base


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
        5. ``json.dumps(..., ensure_ascii=False)`` 以可读中文打印结果。

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

