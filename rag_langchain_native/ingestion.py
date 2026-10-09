"""V3 的 PDF Ingestion（知识库构建）模块。

文件职责：
    完成 ``PDF -> 页面 Document -> Chunk Document -> Embedding -> Chroma``。

为什么需要 Ingestion：
    PDF 是面向阅读和排版的文件，不能直接作为高效检索索引。整份 PDF 也可能超出 LLM
    Context，且每次问答都解析全文成本很高。因此系统先离线提取文本、切成 Chunk、生成
    Embedding 并持久化；在线 Query 只需检索少量相关 Chunk。

输入与输出：
    输入是 PDF 路径和 ``Settings``；输出是 ``Document`` 列表或入库统计字典。
    Metadata 从本模块开始携带 source、1-based page、chunk_id、document_id，并一直保留
    到 Retrieval、Reranker 与 Citation。

调用关系：
    上游是 CLI ``ingest``、FastAPI ``/upload`` 和 V3 Retrieval Evaluation 的可选重建。
    下游是 ``PyMuPDFLoader``、``RecursiveCharacterTextSplitter``、VectorStore。

LangChain 组件：
    ``PyMuPDFLoader``、``Document``、``RecursiveCharacterTextSplitter``、Chroma。
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import DEFAULT_SETTINGS, Settings
from .vectorstore import add_documents, create_vectorstore


def _stable_id(source: str, page: int, chunk_id: int) -> str:
    """根据来源、页码和块号生成稳定且紧凑的 Document ID。

    同一个 PDF 使用相同切分参数重复入库时会得到相同 ID，避免无限产生重复 Chunk。
    SHA-256 在这里用于稳定映射，不承担密码安全用途。
    """
    raw = f"{source}|{page}|{chunk_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:24]


def load_pdf_pages(pdf_path: str | Path) -> list[Document]:
    """使用 LangChain PyMuPDFLoader 把 PDF 加载成逐页 Document。

    参数：
        pdf_path (str | Path): PDF 文件路径。类型联合表示两种类型都可以传入。

    返回：
        list[Document]: 每页一个 Document；``page_content`` 是页面文本，``metadata``
        至少包含文件名 source 和从 1 开始的 page。

    执行过程：
        1. ``resolve()`` 得到绝对路径。
        2. ``PyMuPDFLoader(..., mode="page").load()`` 一次性加载所有页面。
        3. Loader 的 page 通常从 0 开始，本项目加 1 以匹配测试集和用户看到的页码。
        4. 创建新的 Document，避免直接修改 Loader 返回对象的 Metadata。

    LangChain 知识点：
        ``load()`` 返回 ``list[Document]``；某些 Loader 还提供 ``lazy_load()`` 逐个 yield，
        适合超大文档。本项目实际使用的是 ``load()``，没有使用 ``lazy_load()``。
    """
    path = Path(pdf_path).resolve()
    # ``mode="page"`` 很关键：若整份 PDF 只生成一个 Document，将无法可靠保留页码 Citation。
    pages = PyMuPDFLoader(str(path), mode="page").load()
    normalized = []
    for page in pages:
        # ``dict(...)`` 复制 Metadata，避免对 Loader 返回对象产生意外的原地修改。
        metadata = dict(page.metadata)
        zero_based_page = int(metadata.get("page", 0))
        metadata.update(
            source=path.name,
            page=zero_based_page + 1,
        )
        normalized.append(
            Document(page_content=page.page_content, metadata=metadata)
        )
    return normalized


def split_pages(
    pages: list[Document],
    settings: Settings = DEFAULT_SETTINGS,
) -> list[Document]:
    """按页切分文本，并为每个 Chunk 补充稳定 Metadata。

    参数：
        pages (list[Document]): ``load_pdf_pages()`` 返回的页面 Document。
        settings (Settings): 提供 chunk_size 与 chunk_overlap。

    返回：
        list[Document]: Chunk Document 列表。文本在 ``page_content``，来源信息在
        ``metadata``，例如 ``source/page/chunk_id/document_id``。

    为什么需要 Chunk 和 Overlap：
        整页可能同时包含多个主题，Embedding 难以准确代表全部内容；小 Chunk 能提高
        检索粒度。Overlap 让相邻 Chunk 共享边界文字，降低关键句刚好被切断的风险。

    执行过程：
        1. 创建递归字符切分器，优先按段落、换行和中文标点寻找边界。
        2. 每次只切一页，保证一个 Chunk 不跨越 PDF 页码。
        3. ``split_documents()`` 会复制原 Document Metadata。
        4. 按页内顺序添加 chunk_id 和稳定 document_id。
    """
    # ``length_function=len`` 表示按 Python 字符数近似控制大小，而不是按模型 Token 数。
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        # 分隔符按优先级排列；最后的空字符串保证极端长文本最终仍可被切开。
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
    )
    chunks: list[Document] = []
    for page in pages:
        page_chunks = splitter.split_documents([page])
        for index, chunk in enumerate(page_chunks):
            metadata = dict(chunk.metadata)
            source = str(metadata["source"])
            page_number = int(metadata["page"])
            metadata.update(
                chunk_id=index,
                document_id=_stable_id(source, page_number, index),
            )
            chunks.append(
                Document(
                    page_content=chunk.page_content,
                    metadata=metadata,
                )
            )
    return chunks


def ingest_pdf(
    pdf_path: str | Path,
    *,
    settings: Settings = DEFAULT_SETTINGS,
    vectorstore=None,
) -> dict:
    """把一个 PDF 切分并写入 V3 Chroma。

    参数：
        pdf_path (str | Path): 要入库的 PDF。
        settings (Settings): V3 配置。``*`` 之后的参数只能用关键字传入。
        vectorstore: 可选的现有 Chroma；批量重建时复用它可避免重复创建对象。

    返回：
        dict: 文件名、页数、Chunk 数和稳定 Document ID 列表。

    执行过程：
        加载页面 -> 切分 -> 按 source 删除该文件旧 Chunk -> 写入新 Chunk。
        先删后写使同名 PDF 更新时不会留下旧版本内容；操作范围仅是 V3 collection。

    调用关系：
        上游是 CLI、API Upload、``rebuild_knowledge_base()``；下游是 Loader、Splitter 和
        ``vectorstore.add_documents()``。
    """
    store = vectorstore or create_vectorstore(settings)
    pages = load_pdf_pages(pdf_path)
    chunks = split_pages(pages, settings)
    source = Path(pdf_path).name
    store.delete(where={"source": source})
    ids = add_documents(store, chunks)
    return {
        "source": source,
        "pages": len(pages),
        "chunks": len(chunks),
        "document_ids": ids,
    }


def _safe_clear_chroma(settings: Settings) -> None:
    """只允许删除 V3 runtime 内的 Chroma 目录。

    这是破坏性操作前的路径护栏。``is_relative_to(runtime)`` 确认目标确实位于 V3
    runtime 下，且不允许把整个 runtime 当作目标，从而保护 V1/V2 数据和其他缓存。
    """
    target = settings.chroma_dir.resolve()
    runtime = settings.runtime_dir.resolve()
    if not target.is_relative_to(runtime) or target == runtime:
        raise ValueError(f"Refusing to delete non-V3 path: {target}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)


def rebuild_knowledge_base(
    pdf_dir: str | Path | None = None,
    *,
    settings: Settings = DEFAULT_SETTINGS,
) -> dict:
    """从指定目录的全部 PDF 重建 V3 知识库。

    参数：
        pdf_dir (str | Path | None): PDF 目录；None 时使用 ``settings.source_pdf_dir``。
        settings (Settings): V3 配置。

    返回：
        dict: 文件数、总 Chunk 数、逐文件详情和最终 collection 条数。

    执行过程：
        安全清空 V3 Chroma -> 创建空 VectorStore -> 按文件名排序遍历 ``*.pdf`` ->
        复用同一个 store 调 ``ingest_pdf()`` -> 汇总统计。

    初学者知识点：
        列表推导式把“遍历每个 PDF 并返回结果”的循环写成列表；它不是异步执行。
    """
    _safe_clear_chroma(settings)
    store = create_vectorstore(settings)
    directory = Path(pdf_dir or settings.source_pdf_dir)
    results = [
        ingest_pdf(path, settings=settings, vectorstore=store)
        for path in sorted(directory.glob("*.pdf"))
    ]
    return {
        "files": len(results),
        "chunks": sum(item["chunks"] for item in results),
        "details": results,
        "collection_count": store._collection.count(),
    }

