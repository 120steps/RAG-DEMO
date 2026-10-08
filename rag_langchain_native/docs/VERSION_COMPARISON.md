# V1 / V2 / V3 真实代码对比

## 结论

V3 是 LangChain-first，但不是“所有算法零自定义”。Loader、Splitter、Embedding 接口、Chroma、
BM25、RRF 基类、Cross-Encoder compressor、Prompt、ChatModel 和总编排由 LangChain 提供；稳定
ID、中文分词、评分可观测性、多 Query 二次融合、Guard、Citation 和 Evaluation 仍是项目代码。

## 逐项对比

| 能力 | V1 Manual | V2 Wrapper | V3 Native | 说明 |
|---|---|---|---|---|
| PDF Loader | `document_loader.load_pdf` + PyMuPDF | 未替换 | `PyMuPDFLoader.load` | V3 使用标准 DocumentLoader |
| Chunk | `split_text` 固定窗口 | 未替换 | `RecursiveCharacterTextSplitter` | 大小相同不代表边界相同 |
| Embedding | `embedding.embed_text(s)` | `ExistingEmbeddingAdapter` 包旧函数 | `HuggingFaceEmbeddings` | V3 正确区分 E5 query/passage prefix |
| Vector Store | raw `chromadb.Collection` | 独立实验 Chroma | `langchain_chroma.Chroma` | V3 业务链直接使用 VectorStore |
| Retriever | 手写 `collection.query` | 高级链仍调旧 retrieval | `BaseRetriever` / `as_retriever` | V3 可 invoke/batch/ainvoke |
| BM25 | 自写公式 | 复用 V1 | `BM25Retriever` + jieba | 中文预处理仍自定义 |
| Hybrid/RRF | `merge_candidates` / `fuse_results` | 复用 V1 | `EnsembleRetriever` + trace 子类 | V3 用 stable ID，保留 RRF score |
| Reranker | Sentence Transformers `CrossEncoder` | 复用 V1 | LangChain CrossEncoder compressor | V3 扩展 score metadata |
| Rewrite | google-genai 手写调用 | 复用 V1 | Prompt + ChatModel + parser | V3 Cache 独立 |
| Expansion | google-genai 手写调用 | 复用 V1 | Prompt + structured ChatModel | 多 Query 用 Retriever.batch + RRF |
| Prompt | `rag_service.py` f-string | V2 有 ChatPromptTemplate | V3 `ChatPromptTemplate` | V3 是主链组件 |
| LLM | `client.models.generate_content` | ChatGoogleGenerativeAI | ChatGoogleGenerativeAI structured | V3 refusal 是字段，不是字符串猜测 |
| Citation | 手写 metadata | 手写/适配 | Document.metadata -> DTO | 三者都应保留确定性来源 |
| Evaluation | 根 `eval/` | 根 benchmark | V3 独立 `eval/` | Ground Truth 仍只读同一文件 |
| 编排 | 一个 Python 服务函数 | LCEL 包装部分旧阶段 | LCEL Sequence/Parallel/Branch | V3 的数据流由 Runnable 实际执行 |
| Async/Batch | 无统一接口 | 框架链可用但业务未接入 | `aask_rag` / `batch_ask` | 已有接口和离线测试 |
| Streaming | 无 | 未使用 | 未对外实现 | 不夸大框架能力 |

## Before / After 例子

### 1. PDF

V1：

```python
doc = pymupdf.open(file_path)
for page_index, page in enumerate(doc):
    text = page.get_text("text", sort=True).strip()
```

V3：

```python
pages = PyMuPDFLoader(str(path), mode="page").load()
```

Loader 省掉资源遍历样板，但页码契约仍由 V3 校正。

### 2. Embedding

V1：

```python
embedding_model.encode(text).tolist()
```

V2：

```python
class ExistingEmbeddingAdapter(Embeddings):
    # 调用旧 embed_text(s)
```

V3：

```python
HuggingFaceEmbeddings(
    encode_kwargs={"prompt": "passage: "},
    query_encode_kwargs={"prompt": "query: "},
)
```

V2 主要统一接口；V3 的模型生命周期和编码规范归 LangChain component 管理。

### 3. BM25

V1 的 `BM25Index` 手写 term frequency、IDF 和长度归一化。V3：

```python
BM25Retriever.from_documents(docs, preprocess_func=chinese_tokenize, k=10)
```

公式代码消失，但 `jieba` 分词和语料刷新时机仍需设计。

### 4. RAG 编排

V1 的 `rag_service.ask_rag()` 依次修改普通局部变量。V2 的高级 retrieval 仍调用旧
`retrieve_candidates`。V3：

```python
normalize | RunnableParallel(...) | retrieve | rerank | guard | RunnableBranch(...)
```

V3 的实际阶段边界、分支和输入输出出现在 LCEL 图中，可统一 invoke/batch/ainvoke。

### 5. Refusal

V1 后期增加检索阈值和 `refused`。V3 既保留 deterministic guard，又使用：

```python
model.with_structured_output(AnswerOutput)
```

这解决“答案文字已经拒答，但结构化字段仍是 false”的接口不一致；它并不保证模型永不判断错。

## 真实 Benchmark（非严格公平）

V1/V2 来自既有 `eval/results/langchain_full_benchmark.json`；V3 是本次新跑。V3 使用不同的
Splitter、E5 前缀和归一化，因此只能做架构回归参考，不能把差异全部归因于 LangChain。

| 指标 | V1 历史 | V2 历史 | V3 本次 |
|---|---:|---:|---:|
| Retrieval Hit@1 | 100% | 100% | 96.43% |
| Hit@3/5/10 | 100% | 100% | 100% |
| Deterministic Answer | 89.29% | 89.29% | 待完整重跑 |
| Citation | 100% | 100% | 待完整重跑 |
| Refusal | 100% | 100% | 待完整重跑 |
| Judge Correctness / 2 | 1.906 | 1.906 | 待完整重跑 |
| Judge Faithfulness / 2 | 2.000 | 1.969 | 待完整重跑 |
| Hallucination | 0% | 3.12% | 待完整重跑 |
| 平均端到端延迟 | 6832 ms | 11361 ms | 待完整重跑 |

V3 首轮完整评估后修改了 structured refusal 契约；选择性重跑只能诊断修复，不能作为最终
全量结果。完整 `--force` 重跑的外部调用授权被拒绝，因此结果文件明确标为无效，文档不引用
混合数据。不同时间、不同缓存状态下的延迟也不应当作严格性能结论。

## V3 消融

| 模式 | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR |
|---|---:|---:|---:|---:|---:|
| Vector Only | 89.29% | 100% | 100% | 100% | 0.9464 |
| Hybrid | 92.86% | 100% | 100% | 100% | 0.9643 |
| Hybrid + Reranker | 96.43% | 100% | 100% | 100% | 0.9821 |

Rewrite 两组与 Rewrite+Expansion 组没有实际运行：对应外部 Gemini 执行授权被拒绝，结果中
标记为 skipped，不能给出分数。

## LangChain-first 的真实收益与代价

收益是统一 `Document/Retriever/Runnable` 契约、可替换组件、LCEL 可读编排、统一 batch/async、
结构化输出和生态集成。代价是依赖显著增加、Classic API 仍有版本迁移风险、metadata 评分仍需
扩展、抽象栈让调试更深。LangChain 不会自动提高 Recall、不会自动消除幻觉，也不一定更快。

