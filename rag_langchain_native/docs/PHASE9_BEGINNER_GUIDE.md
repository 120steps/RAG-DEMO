# Phase 9 初学者代码导读

## 1. 先理解四个 ID

- `tenant_id`：哪家企业。
- `knowledge_base_id`：该企业里的哪套知识库。
- `document_id`：逻辑文档，更新文件时保持不变。
- `version_id`：某次不可变文件内容。
- `chunk_uid`：某版本某页某块的唯一向量记录。

如果只使用文件名做 ID，Tenant A 和 Tenant B 都上传 `policy.pdf` 就会冲突；如果把版本
覆盖掉，就无法回滚，也无法说明 Citation 指向的是哪一版。

## 2. 推荐阅读顺序

1. `config.py::Settings`：先看路径、collection、模型和 Phase 9 开关。
2. `catalog.py::DocumentCatalog.initialize()`：理解 SQLite 保存什么。
3. `lifecycle.py::index_version()`：跟踪 PDF 怎样变成企业 Chunk。
4. `security.py::AuthService` 和 `build_authorization_scope()`：区分认证与授权。
5. `retrieval.py::NativeRetrievalEngine`：看 Vector/BM25 如何使用同一范围。
6. `enterprise.py::EnterpriseRAGService.chat()`：看企业主流程。
7. `conversation.py::contextualize()` 与 `router.py`：看多轮和分流。
8. `api.py::create_app()`：最后看 HTTP 如何调用上面所有服务。

## 3. SQLite 与 Chroma 为什么都需要

SQLite 擅长关系和状态：用户属于哪个租户、文档当前是哪一版、谁有 ACL。Chroma 擅长
“哪个 Chunk 与 Query 的向量最相似”。让 Chroma 单独承担版本工作流会很别扭；让 SQLite
做向量近邻搜索也不合适。

## 4. Register、Upload、Index、Publish

```text
register_document -> 只有逻辑文档
upload_version    -> 文件已保存，状态 registered
index_version     -> Loader / Splitter / Embedding / Chroma，状态 indexed
publish_version   -> active_version_id 指向该版本
```

“索引”和“发布”分开是安全设计。新 PDF 即使解析失败，用户继续检索旧 published 版本。

## 5. Authentication 与 Authorization

Authentication 是“确认 Alice 真的是 Alice”；Authorization 是“Alice 能不能看 HR 文档”。

`AuthService.verify_token()` 不信任请求里的 role。它验证 Token 签名后，用 `user_id` 回
SQLite 重读 tenant/roles/groups。`build_authorization_scope()` 再把这些信息与 Document
classification/ACL 组合成允许的 `version_ids`。

RBAC 是按角色授权，例如 HR 角色读 HR 文档；ACL 是某一份文档的例外规则，例如把一份
HR 文档额外授予某个 user 或 group。

## 6. 为什么必须在 Retrieval 阶段过滤

如果先全库检索、最后才删除未授权文档，未授权文本已经进入 Reranker、日志，甚至可能
进入 LLM Context。当前代码让：

- Chroma 查询直接带 `version_id in allowed_versions`；
- BM25 只从 allowed version 构建索引；
- RRF/Reranker 只接收两者的授权候选；
- 输出前 `_verify_output_scope()` 再检查一次。

空权限集合必须直接返回空结果，绝不能因 filter 为空而调用全库搜索。

## 7. Multi-Tenant 不只是多一个字段

除了 Metadata 里的 `tenant_id`，还需要隔离 Catalog 查询、上传目录、BM25 语料、Cache
Key、Conversation 所有权和 Citation。当前实现都做了逻辑隔离，但共享 Chroma collection，
所以仍是 Enterprise Prototype，而非最高等级物理隔离。

## 8. Conversation 如何处理“那谁批准？”

`ConversationService.contextualize()` 读取当前用户最近历史，把：

```text
历史：国际出差提前多久申请？
本轮：那谁批准？
```

改写成类似“国际出差申请最终由谁批准？”。这是 Retrieval Query。最后 Answer Prompt 的
Question 仍是“那谁批准？”，避免把模型改写冒充成用户原话。模型不可用时回退原问题。

## 9. Router 不是 Agent

`QueryRouter` 是一个 `RunnableLambda`。明确问候走固定 `normal_chat`，不读取知识库；其他
输入默认走 `knowledge_rag`。它不选择工具、不循环规划，也不拥有额外权限，所以不是 Agent。

## 10. Cache Invalidation

Phase 9 实际存在的是 Rewrite/Expansion JSON Cache，没有 Retrieval/Answer Cache。企业
Cache Key 加入 tenant、kb、user 权限摘要、allowed versions 和 `epoch`。发布、回滚或删除
使 epoch 增加，即使旧文件仍在磁盘，也不会命中新状态的 Key。

## 11. 十个关键方法

1. `DocumentCatalog.transaction()`：异常 rollback，正常 commit，并关闭连接。
2. `register_document()`：创建稳定逻辑 Document。
3. `create_version()`：用 content hash 实现版本幂等。
4. `DocumentLifecycleService.index_version()`：生成企业 Metadata 并写 Chroma。
5. `publish_version()`：切换 active pointer，不覆盖旧向量。
6. `AuthService.verify_token()`：验证身份并重载权限。
7. `build_authorization_scope()`：算出唯一可信检索范围。
8. `NativeRetrievalEngine.retrieve_queries()`：授权范围内 Hybrid + 多 Query RRF。
9. `ConversationService.contextualize()`：历史 + 当前问题的 LCEL 改写。
10. `EnterpriseRAGService.chat()`：把 Router、Scope、LCEL RAG、Citation、Message 串起来。

## 12. 哪些是 LangChain，哪些不是

LangChain：`Document`、PyMuPDFLoader、TextSplitter、Embeddings、Chroma VectorStore、
BM25Retriever、EnsembleRetriever、CrossEncoderReranker、ChatPromptTemplate、ChatModel、
Runnable/LCEL、OutputParser。

普通 Python/平台代码：SQLite Schema/事务、HMAC Token、PBKDF2、RBAC/ACL、文件路径、版本
状态机、FastAPI Dependencies、Cache Namespace。LangChain 不会自动解决企业权限和版本。

## 13. 常见误区

1. “Token 能解码，所以可信”是错的；必须验证签名和过期时间。
2. “Vector 过滤了，BM25 不过滤也没事”是错的；Fusion 会把 BM25 越权结果带回来。
3. `document_id` 不是 Chunk ID；Phase 9 必须用 `chunk_uid` 去重。
4. Published 不是“向量已经写入”的同义词；先 indexed，再显式 publish。
5. Hit@K 很高不代表权限安全；必须有恶意/跨租户负向测试。
6. LLM Judge 2 分不代表结构化字段一定一致；Q023 的正文与 refused 标志就是反例。

## 14. 动手练习

1. 在测试里创建 Tenant A/B 同名 PDF，观察两个 `chunk_uid` 为什么不同。
2. 给 general 用户查询 HR 文档，断点查看 `AuthorizationScope.version_ids`。
3. 发布 V2 再回滚 V1，比较 Citation 的 `version_id`。
4. 篡改 Token 最后一个字符，跟踪 `verify_token()` 为什么拒绝。
5. 给 HR 文档增加 user ACL，再比较 Scope 前后的 document_ids。

