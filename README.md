# 企业知识库 RAG Demo

这是一个用于学习和验证企业知识库 RAG 的 Python 项目。项目同时保留两套实现：

- 手写 RAG：当前 FastAPI 业务入口，包含 Hybrid Search、RRF、Reranker、Query Rewrite、Query Expansion、拒答和 Citation。
- LangChain RAG：与手写实现并行，用 LangChain Runnable 编排现有高级 Retrieval，并使用 `ChatGoogleGenerativeAI` 生成答案。

LangChain 版本没有重写已经验证过的 BM25、RRF、Reranker、Rewrite 和 Expansion，而是通过适配器和 Runnable 复用它们。

详细代码讲解见 [Phase 8 LangChain 代码指南](docs/PHASE8_LANGCHAIN_CODE_GUIDE.md)。

## 当前架构

```text
PDF
  -> PyMuPDF 解析
  -> 手写 Chunk
  -> multilingual-e5-base Embedding
  -> ChromaDB

Question
  -> Query Rewrite（可选）
  -> Query Expansion（可选）
  -> Vector + BM25 Hybrid Retrieval
  -> RRF（多 Query 时）
  -> CrossEncoder Reranker
  -> Answerability Guard
  -> Context + Prompt
  -> Gemini
  -> Answer + Metadata Citation
```

当前 FastAPI `/chat` 调用 `rag_service.ask_rag()`。LangChain 完整实现的函数入口是 `langchain_rag.lc_full_rag.ask_langchain_full_rag()`；它尚未替换 FastAPI 的业务入口。

## 主要目录

```text
rag-demo/
├── app.py                         # FastAPI，当前调用手写 RAG
├── rag_service.py                 # 手写完整 RAG
├── retrieval.py                   # Vector/Hybrid/多 Query Retrieval 编排
├── bm25_search.py                 # BM25 与候选合并
├── result_fusion.py               # RRF
├── reranker.py                    # 本地 CrossEncoder Reranker
├── query_rewriter.py              # Query Rewrite 与本地 Cache
├── query_expander.py              # Query Expansion 与本地 Cache
├── answer_guard.py                # 结构化拒答判断
├── embedding.py                   # 原 Embedding 实现
├── document_loader.py             # PDF 解析和手写 Chunk
├── ingest.py                      # 原 Chroma 入库脚本
├── langchain_rag/
│   ├── lc_embedding.py            # LangChain Embeddings Adapter
│   ├── lc_vectorstore.py          # 独立 LangChain Chroma VectorStore
│   ├── lc_retriever.py            # 纯向量 Retriever
│   ├── lc_llm.py                  # ChatGoogleGenerativeAI
│   ├── lc_rag.py                  # 最小纯向量 LangChain RAG
│   └── lc_full_rag.py             # 完整高级 LangChain RAG
├── eval/
│   ├── test_case.json             # 唯一真实评估集
│   ├── retrieve_eval.py           # Retrieval Evaluation
│   ├── answer_eval.py             # Deterministic Answer Evaluation
│   ├── answer_judge.py            # LLM-as-a-Judge
│   ├── end_to_end_eval.py         # 手写 RAG 端到端评估
│   ├── experiment_langchain.py    # 纯 Vector 回归对比
│   └── experiment_langchain_full.py # 完整 RAG 对比
├── data/pdf/                      # 原始 PDF
├── chroma_db/                     # 原业务向量库（Git 忽略）
├── chroma_db_langchain/           # LangChain 实验向量库（Git 忽略）
├── requirements.txt               # 原项目依赖
└── phase8-requirements.txt        # Phase 8 独立依赖
```

## 环境

原项目环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Phase 8 LangChain 环境：

```powershell
python -m venv .venv-phase8
.\.venv-phase8\Scripts\Activate.ps1
pip install -r phase8-requirements.txt
pip check
```

在项目根目录的 `.env` 中配置：

```env
GEMINI_API_KEY=your_api_key
```

API Key 由 `config.py` 加载，不要写入源码或提交到 Git。

## 配置

主要配置位于 `config.py`：

- `LLM_MODEL`
- `CHUNK_SIZE` / `CHUNK_OVERLAP`
- `TOP_K`
- `QUERY_REWRITE_ENABLED`
- `QUERY_EXPANSION_ENABLED`
- Answerability Guard 阈值

Embedding 模型由 `embedding.py` 定义。目前使用 `intfloat/multilingual-e5-base`，原实现直接编码原始文本，没有添加 `query:` 或 `passage:` 前缀。

## 构建知识库

```powershell
python ingest.py
```

注意：`ingest.py` 会删除并重建原 `company_knowledge` collection。只有在明确需要重新切 Chunk 和生成 Embedding 时才运行。

LangChain 独立 VectorStore 位于 `chroma_db_langchain/`，不会删除或覆盖原 `chroma_db/`。

## 运行入口

### 手写命令行 RAG

```powershell
python rag.py
```

### FastAPI

```powershell
uvicorn app:app --reload
```

主要接口：

- `GET /health`
- `POST /chat`
- `POST /documents`
- `GET /documents`
- `DELETE /documents/{filename}`

### LangChain 完整 RAG

Python 调用：

```python
from langchain_rag.lc_full_rag import ask_langchain_full_rag

result = ask_langchain_full_rag("国际出差需要提前多久申请？")
print(result["answer"])
print(result["sources"])
```

该调用会使用 Gemini，可能产生费用。最终答案回答原始问题；Rewrite/Expansion Query 只用于 Retrieval。

## Evaluation

以下命令在项目根目录运行。

纯 Retrieval，不调用最终 Answer Generation：

```powershell
python eval/retrieve_eval.py
```

Deterministic Answer Evaluation（调用 Gemini）：

```powershell
python eval/answer_eval.py
```

手写 RAG End-to-End Evaluation（调用 Gemini 和 Judge）：

```powershell
python eval/end_to_end_eval.py
```

Manual Vector 与 LangChain Vector 回归：

```powershell
.\.venv-phase8\Scripts\python.exe eval\experiment_langchain.py
```

Manual Full RAG 与 LangChain Full RAG Benchmark：

```powershell
.\.venv-phase8\Scripts\python.exe eval\experiment_langchain_full.py
```

完整 RAG Benchmark 会产生较多 Gemini 请求；已有结果位于 `eval/results/langchain_full_benchmark.json`。除非配置或实现发生变化，否则无需频繁重跑。

## 已验证的 Phase 8 结论

- Embedding Adapter 与原 Embedding 数值完全一致。
- 原 Chroma 与 LangChain 独立 Chroma 的 93 个 Chunk、Metadata 和稳定 ID 一致。
- 纯 Vector Regression 的 Top-10 顺序与 L2 Distance 完全一致。
- 完整 Retrieval Benchmark 中，Manual 与 LangChain 的 Hit@1/3/5/10 都是 100%。
- LangChain 提供统一接口和编排能力，但不会自动提升 Retrieval Accuracy 或消除 Hallucination。

## 安全边界

- `eval/test_case.json` 是 Ground Truth，不应为提高指标而修改。
- `chroma_db/` 是原业务库；LangChain 实验使用独立数据库。
- Citation 必须来自 Retrieved Metadata，不应信任模型自行生成的来源。
- Query Rewrite/Expansion 失败时应回退到原 Query，不能让整个 RAG 请求失败。
- `.env`、虚拟环境和本地 Chroma 数据库均被 Git 忽略。
