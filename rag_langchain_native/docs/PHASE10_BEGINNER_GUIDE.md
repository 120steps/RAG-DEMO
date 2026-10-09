# Phase 10 初学者指南：一次 RAG 请求如何被观察

## 1. 可观测性不是 `print()`

`print()` 很难区分并发请求，也不能可靠计算 P95。Phase 10 使用四个概念：

- Trace：一次请求的完整旅程。
- Span：旅程中的一个阶段，例如 Vector Search。
- Metrics：跨很多请求聚合的计数和时延。
- Structured Log：可被程序解析的 JSON 事件。

LangChain Callback 比散落的 `print()` 更合适：LCEL、Retriever、ChatModel 都会发出统一 start/end/error 事件，并携带 `run_id` / `parent_run_id`，可以还原父子关系。

## 2. 推荐阅读顺序

1. `observability/config.py`：默认隐私和本地模式。
2. `observability/core.py::request_scope()`：请求如何获得 request_id/trace_id。
3. `observability/core.py::stage()`：普通 Python 阶段如何成为 Span。
4. `observability/callbacks.py::ObservabilityCallback`：LangChain 如何通知生命周期。
5. `api.py::observability_middleware()`：HTTP 如何接入根 Trace。
6. `retrieval.py`、`reranker.py`：算法旁的真实阶段埋点。
7. `chain.py::ask_rag()`：CLI/Evaluation 如何也有 Trace。
8. `observability/report.py`：如何生成树和统计。

## 3. 十个关键方法

### `request_scope(operation, request_id, tenant_id)`

进入 `with` 创建 `rag.request`，离开时无论成功或异常都写摘要。tenant_id 只变成不可逆短指纹。

### `stage(operation, attributes)`

把真实执行放在 `with manager.stage("vector.search")`。成功记录时延/数量；异常分类后仍把原异常交给原业务逻辑，不吞错。

### `ContextVar`

它像“当前异步任务自己的变量”。两个并发请求看到不同 `RequestState`，不会串 request_id。它不做认证；认证仍由 `AuthService` 完成。

### `ObservabilityCallback`

`chain.invoke(..., config={"callbacks": [...]})` 触发 `on_chain_start/end`、`on_retriever_start/end`、`on_chat_model_start`、`on_llm_end`。Callback 不读取 Prompt/正文，只取名称、数量、时延和 Usage。

### `extract_token_usage()`

输入 `LLMResult`，查找真实 Usage；输出 `{available, input_tokens, output_tokens, total_tokens}`。找不到就 unavailable。

### `PricingCatalog.estimate()`

真实 Usage 与管理员核实单价同时存在才计算。结果是 estimate，不是 invoice。

### `redact()`

写 JSON 前递归清理字典/列表。凭证和业务正文变成 `[REDACTED]`，保护集中在一处。

### `bind_tenant()`

认证成功后把租户指纹绑定到当前请求。不能在认证前信任客户端 tenant，也不能用该指纹代替授权。

### `LocalReportStore.trace_detail()`

先根据 request_id/trace_id 找请求，再按 `parent_span_id` 生成树。带 tenant scope 时先验证归属。

### `LocalReportStore.summary()`

在明确时间窗口内计算样本量、错误率、平均/P50/P95、RPM、Token、估算费用和常见错误。小样本 P95 不代表生产性能。

## 4. 一次请求的数据流

```text
POST /chat
  -> middleware 创建 request_id 与 rag.request
  -> current_principal 验证 Token，绑定 tenant 指纹
  -> EnterpriseRAGService 记录 route / authorization
  -> LCEL Callback 记录 Chain / ChatModel
  -> Retriever 埋点记录 Vector / BM25 / RRF
  -> Reranker / Context / Citation
  -> 请求结束写 requests.jsonl
  -> 响应返回 X-Request-ID / X-Trace-ID
```

请求摘要用于“最近请求和聚合”；Span 用于“这一条内部发生了什么”；Metrics 用于“长期趋势”。三者不能互相完全替代。

## 5. 为什么 Vector、BM25、RRF 分别追踪

Hybrid 失败可能是 Vector 慢、BM25 零候选、Fusion 输入异常或 Reranker 降错序。只有总 Retrieval 时延无法定位。系统记录各阶段与数量，但不记录候选正文，避免 ACL 内容泄露。

## 6. Fallback、错误和拒答

Rewrite 超时后用原 Query 继续，是“局部降级”：有 failure/fallback，最终根请求仍可能 success。认证失败或最终 LLM 失败导致无法完成才是请求错误。知识不足而拒答是业务结果，不是技术故障。

## 7. 常见误区

1. Trace ID 不是访问密码；读 Trace 仍需认证、admin 和租户校验。
2. Metrics Label 不放 request_id/user_id，否则产生高基数和隐私风险。
3. Token 不能按字符数伪造；SDK 没给就 unavailable。
4. 估算费用不是账单，价格必须有版本。
5. Span 越多不一定越好，只跟踪真实且可行动的阶段。
6. Observability 开启不意味着可以写全文，默认最小采集更安全。
7. OTel 自动 FastAPI Span 与业务 `rag.request` 可以同时存在：前者描述 HTTP，后者描述 RAG 业务。

## 8. 自测题与答案

1. request_id 与 trace_id 区别？应用关联 ID vs OpenTelemetry 整棵 Span 树 ID。
2. 为什么用 ContextVar？防止并发上下文串线。
3. Rewrite 失败但回答成功算什么？局部降级，不是整次失败。
4. 为什么 tenant_id 不做 Metric Label？高基数与隐私风险。
5. Callback 为什么不读 Prompt？生命周期追踪不需要正文，最小采集更安全。
6. Usage 缺失怎么办？unavailable，不估算 Token/费用。
7. Reranker 为何记录输入输出数？定位候选池与截断。
8. 本地同步 JSONL 的局限？吞吐较低，生产宜用 Batch OTLP。
9. 正常拒答是否增加 error count？不增加，它是业务结果。
10. 为什么 Trace 查询还要 ACL？Trace 可能暴露内部结构，ID 本身不是授权。
