"""PowerShell-friendly command line interface for V3."""

from __future__ import annotations

import argparse
import json
import sys

from .chain import get_service
from .config import DEFAULT_SETTINGS
from .ingestion import ingest_pdf, rebuild_knowledge_base


def build_parser() -> argparse.ArgumentParser:
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
    raise SystemExit(main())

