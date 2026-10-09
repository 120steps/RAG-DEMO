"""LangChain V3 的集中配置模块。

文件职责：
    统一保存 V3 的目录、模型、Chunk、Retrieval、Reranker、Query Processing、
    Gemini 和拒答阈值配置。V3 不读取 V1 的 ``config.py``，因此可以独立运行和实验。

在 RAG Pipeline 中的位置：
    配置本身不处理文档或问题，而是被 Ingestion、Embedding、Retrieval、Reranker、
    Query Processing、Answer Chain、CLI、API 和 Evaluation 共同读取。

输入与输出：
    输入主要来自环境变量；没有环境变量时使用代码中明确写出的默认值。
    输出是不可变的 ``Settings`` 对象。调用方通过属性读取配置，例如
    ``settings.chunk_size`` 或 ``settings.vector_k``。

设计原因：
    如果每个模块各自硬编码模型名称和 Top-K，很容易出现“入库参数”和“查询参数”
    不一致。集中配置让实验条件可追踪，也防止 V3 误写 V1/V2 的运行目录。

LangChain 关系：
    本文件没有直接创建 LangChain 组件；它为 ``HuggingFaceEmbeddings``、``Chroma``、
    Retriever、Cross-Encoder 和 ``ChatGoogleGenerativeAI`` 提供构造参数。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _bool_env(name: str, default: bool) -> bool:
    """把环境变量转换为布尔值。

    参数：
        name (str): 环境变量名，例如 ``V3_BM25_ENABLED``。
        default (bool): 环境变量不存在时使用的默认值。

    返回：
        bool: ``1/true/yes/on``（忽略大小写）返回 True，其他已设置值返回 False。

    初学者知识点：
        ``str | None`` 是 Python 的类型联合，表示一个值可能是字符串，也可能是 None。
        这里的 ``|`` 属于类型注解，不是 LCEL 中连接 Runnable 的管道操作符。
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """V3 的不可变配置对象。

    ``@dataclass`` 是装饰器：Python 会根据下面的字段自动生成构造函数等方法。
    ``frozen=True`` 表示实例创建后不能直接修改字段，避免一次实验运行到一半时参数被
    意外改变。需要做消融实验时，Evaluation 使用 ``dataclasses.replace`` 创建新对象。

    路径字段全部指向 V3 自己的 ``runtime``，只有 ``source_pdf_dir`` 只读指向仓库的
    原始 PDF。模型密钥仍从根 ``.env`` 加载，但不会写回文件。
    """

    # ``Path`` 比手工拼接字符串更安全，也能在 Windows 与其他系统上使用正确分隔符。
    package_dir: Path = PACKAGE_DIR
    project_root: Path = PROJECT_ROOT
    source_pdf_dir: Path = PROJECT_ROOT / "data" / "pdf"
    runtime_dir: Path = PACKAGE_DIR / "runtime"
    chroma_dir: Path = PACKAGE_DIR / "runtime" / "chroma"
    cache_dir: Path = PACKAGE_DIR / "runtime" / "cache"
    upload_dir: Path = PACKAGE_DIR / "runtime" / "uploads"
    catalog_path: Path = PACKAGE_DIR / "runtime" / "catalog.sqlite3"

    # Phase 9 使用独立 collection，避免企业版生命周期数据与 Phase 8 基线互相污染。
    enterprise_collection_name: str = os.getenv(
        "V3_ENTERPRISE_CHROMA_COLLECTION", "company_knowledge_enterprise"
    )
    default_knowledge_base_id: str = os.getenv(
        "V3_DEFAULT_KNOWLEDGE_BASE_ID", "default"
    )

    # Authentication：密钥只能来自环境变量。未配置时企业 API 会默认拒绝受保护请求，
    # 而不是使用源码中的弱默认值。测试通过 dataclasses.replace 注入临时密钥。
    auth_secret: str | None = os.getenv("V3_AUTH_SECRET")
    auth_token_ttl_seconds: int = int(
        os.getenv("V3_AUTH_TOKEN_TTL_SECONDS", "3600")
    )

    # Conversation / Router。History 只取最近若干条，避免 Prompt 无限增长。
    conversation_history_limit: int = int(
        os.getenv("V3_CONVERSATION_HISTORY_LIMIT", "10")
    )
    query_router_enabled: bool = _bool_env("V3_QUERY_ROUTER_ENABLED", True)

    # Ingestion / Chroma：Chunk 参数改变后必须重建 V3 自己的向量库。
    collection_name: str = os.getenv(
        "V3_CHROMA_COLLECTION", "company_knowledge_native"
    )
    chunk_size: int = int(os.getenv("V3_CHUNK_SIZE", "200"))
    chunk_overlap: int = int(os.getenv("V3_CHUNK_OVERLAP", "30"))

    # Embedding：V3 使用 E5 的 Query/Passage 角色前缀，并对向量做归一化。
    embedding_model: str = os.getenv(
        "V3_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"
    )
    embedding_device: str = os.getenv("V3_EMBEDDING_DEVICE", "cpu")
    embedding_batch_size: int = int(
        os.getenv("V3_EMBEDDING_BATCH_SIZE", "32")
    )
    normalize_embeddings: bool = _bool_env(
        "V3_NORMALIZE_EMBEDDINGS", True
    )
    query_prefix: str = "query: "
    document_prefix: str = "passage: "

    # Retrieval：先从 Vector/BM25 各取候选，再融合成 candidate_k 个候选。
    vector_enabled: bool = _bool_env("V3_VECTOR_ENABLED", True)
    bm25_enabled: bool = _bool_env("V3_BM25_ENABLED", True)
    vector_k: int = int(os.getenv("V3_VECTOR_K", "10"))
    bm25_k: int = int(os.getenv("V3_BM25_K", "10"))
    candidate_k: int = int(os.getenv("V3_CANDIDATE_K", "20"))
    final_k: int = int(os.getenv("V3_FINAL_K", "3"))
    vector_weight: float = float(os.getenv("V3_VECTOR_WEIGHT", "0.5"))
    bm25_weight: float = float(os.getenv("V3_BM25_WEIGHT", "0.5"))
    rrf_k: int = int(os.getenv("V3_RRF_K", "60"))

    # Reranker：只重排候选池，最终保留 final_k 个 Document。
    reranker_enabled: bool = _bool_env("V3_RERANKER_ENABLED", True)
    reranker_model: str = os.getenv(
        "V3_RERANKER_MODEL",
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    )

    # Query Processing：默认关闭，确保不额外调用 Gemini，也保持基础检索行为稳定。
    rewrite_enabled: bool = _bool_env("V3_QUERY_REWRITE_ENABLED", False)
    expansion_enabled: bool = _bool_env(
        "V3_QUERY_EXPANSION_ENABLED", False
    )
    expansion_count: int = int(os.getenv("V3_QUERY_EXPANSION_COUNT", "3"))
    llm_model: str = os.getenv("V3_LLM_MODEL", "gemini-3.1-flash-lite")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_min_interval_seconds: float = float(
        os.getenv("V3_GEMINI_MIN_INTERVAL_SECONDS", "3.5")
    )

    # Answer Guard：用检索分数判断 Context 是否足以支持回答。
    answerability_guard_enabled: bool = _bool_env(
        "V3_ANSWERABILITY_GUARD_ENABLED", True
    )
    min_rerank_score: float = float(
        os.getenv("V3_MIN_RERANK_SCORE", "2.0")
    )
    min_bm25_score: float = float(os.getenv("V3_MIN_BM25_SCORE", "32.0"))
    max_vector_distance: float = float(
        os.getenv("V3_MAX_VECTOR_DISTANCE", "0.24")
    )
    refusal_message: str = "根据当前知识库无法回答该问题。"

    def ensure_runtime_dirs(self) -> None:
        """确保 V3 运行时目录存在。

        功能：
            创建 Chroma、Cache 和 Upload 所需目录；已存在时不会报错。

        返回：
            None。本函数的结果是文件系统目录被准备好。

        调用关系：
            上游包括 Embedding、VectorStore、Reranker、QueryProcessor 和 API Upload。
            下游是 ``Path.mkdir()``，不会创建或修改 V1/V2 数据目录。

        初学者知识点：
            ``parents=True`` 会创建缺失的父目录；``exist_ok=True`` 允许目录已经存在。
        """
        for directory in (
            self.runtime_dir,
            self.chroma_dir,
            self.cache_dir,
            self.upload_dir,
            self.catalog_path.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = Settings()

