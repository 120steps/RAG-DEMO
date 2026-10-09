# LangChain Native Enterprise RAG V3（Phase 9）

这是与仓库 V1（手写）和 V2（LangChain 包装版）并行的独立实现。V3 不导入旧版
`ask_rag`、Embedding、Hybrid、Reranker、Rewrite 或 Expansion 业务函数。它只读根目录的
`data/pdf/`、`.env` 和 `eval/test_case.json`。

## 环境

```powershell
python -m venv rag_langchain_native/.venv
.\rag_langchain_native\.venv\Scripts\python.exe -m pip install -r rag_langchain_native\requirements.txt
.\rag_langchain_native\.venv\Scripts\python.exe -m pip check
```

已验证环境：Python 3.11.0、LangChain 1.4.3、Core 1.6.7、Community 0.4.1、
Classic 1.0.2、Chroma integration 1.1.0、Google GenAI integration 4.4.0。

## 常用命令

```powershell
# 重建 V3 独立知识库
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli ingest --rebuild

# 只检索
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli retrieve "国外出差领导怎么批？" --top-k 5

# 完整 RAG
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli ask "国外出差领导怎么批？"

# 独立 API（不会占用 V1 端口）
.\rag_langchain_native\.venv\Scripts\python.exe -m uvicorn rag_langchain_native.api:app --host 127.0.0.1 --port 8011

# 测试与评估
.\rag_langchain_native\.venv\Scripts\python.exe -m pytest rag_langchain_native\tests -q
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.retrieval_eval
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.answer_eval
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.ablation --local-only
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.compare_versions
```

API 路由：`GET /health`、`POST /retrieve`、`POST /chat`、`POST /upload`。上传 PDF 和
Chroma、模型缓存、Query Cache 均在 `rag_langchain_native/runtime/`。

## Phase 9 快速启动

Phase 9 新增 SQLite Document Catalog、不可变版本、发布/回滚、签名 Token、RBAC/ACL、
Multi-Tenant、Conversation Memory、Contextual Rewrite 和 Query Router。企业数据使用
`company_knowledge_enterprise` collection，与 Phase 8 的 `company_knowledge_native` 隔离。

先为本地环境设置足够长的随机签名密钥；源码没有默认密钥，未配置时受保护 API 默认拒绝：

```powershell
$env:V3_AUTH_SECRET = "请替换为本地随机长字符串"

# 创建开发管理员（密码参数方式仅适合本地学习环境）
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli create-user `
  --tenant tenant-a --username admin --password "ChangeMe123!" --roles admin

# 上传、索引并发布到独立企业知识库
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli enterprise-upload `
  data\pdf\travel_policy.pdf --tenant tenant-a --username admin `
  --password "ChangeMe123!" --knowledge-base default --classification general --publish

# 权限感知检索/问答
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli enterprise-retrieve `
  "国际出差谁批准？" --tenant tenant-a --username admin --password "ChangeMe123!"
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.cli enterprise-ask `
  "国际出差谁批准？" --tenant tenant-a --username admin --password "ChangeMe123!"
```

API 登录后把返回 Token 放在 `Authorization: Bearer <token>`。主要端点：

- `POST /auth/login`
- `POST/GET/DELETE /conversations...`
- `POST /retrieve`、`POST /chat`
- `POST /documents/register`、`POST /documents/{id}/versions`
- `POST /documents/{id}/publish`、`POST /documents/{id}/rollback`
- `DELETE /documents/{id}`、`POST /documents/{id}/restore`
- `GET /documents/{id}/download`

Phase 9 验证：

```powershell
.\rag_langchain_native\.venv\Scripts\python.exe -m pytest rag_langchain_native\tests -q
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.security_eval
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.enterprise_retrieval_eval --bootstrap
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.enterprise_answer_eval
.\rag_langchain_native\.venv\Scripts\python.exe -m rag_langchain_native.eval.enterprise_judge_eval
```

## 默认配置

- Embedding：`intfloat/multilingual-e5-base`，768 维，CPU，归一化。
- Query / Document 前缀：`query: ` / `passage: `。
- Chunk：200 / overlap 30（边界由 `RecursiveCharacterTextSplitter` 生成，因此不等同 V1）。
- Hybrid：Chroma cosine + `BM25Retriever`（jieba）+ weighted RRF (`c=60`)。
- Reranker：本地 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`。
- Answer：`ChatGoogleGenerativeAI` + structured output。
- Rewrite / Expansion：默认关闭，可通过 `V3_QUERY_REWRITE_ENABLED`、
  `V3_QUERY_EXPANSION_ENABLED` 开启。

详细说明见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)、
[LANGCHAIN_LEARNING_GUIDE.md](docs/LANGCHAIN_LEARNING_GUIDE.md) 和
[VERSION_COMPARISON.md](docs/VERSION_COMPARISON.md)。Phase 9 设计、安全边界和教学说明见
[PHASE9_ENTERPRISE_RAG.md](docs/PHASE9_ENTERPRISE_RAG.md) 与
[PHASE9_BEGINNER_GUIDE.md](docs/PHASE9_BEGINNER_GUIDE.md)。

