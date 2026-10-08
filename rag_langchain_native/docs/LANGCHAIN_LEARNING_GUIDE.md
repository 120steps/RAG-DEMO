# LangChain 学习指南（结合 V3 真实代码）

## 先理解五个数据层次

`Document` 是“文本 + metadata”；Loader 负责从文件生成 Document；Splitter 负责把大
Document 变小但继续携带 metadata；Embeddings 把文本变成向量；Retriever 接收 Query，
返回 `list[Document]`。LCEL 再把这些组件的输入输出连接起来。

## 1. `PyMuPDFLoader.load()`

**位置：** `ingestion.py:load_pdf_pages`

```python
pages = PyMuPDFLoader(str(path), mode="page").load()
```

输入是 PDF 路径，输出是每页一个 `Document`。手写版需要自己打开 PyMuPDF、遍历页、提取
文字和关闭文件；Loader 统一了这部分接口。V3 仍需自己把 0-based `page` 转成 1-based。

## 2. `RecursiveCharacterTextSplitter.split_documents()`

**位置：** `ingestion.py:split_pages`

```python
page_chunks = splitter.split_documents([page])
```

输入 `list[Document]`，输出更小的 `list[Document]`。它会按分隔符优先级递归寻找边界，
不是 V1 那种固定字符窗口；这也是 V1/V3 Chunk 不能声称完全一致的原因。最大收益是 metadata
自动复制到 Chunk。

## 3. `HuggingFaceEmbeddings`

**位置：** `embedding.py:_cached_embeddings`

```python
HuggingFaceEmbeddings(
    model="intfloat/multilingual-e5-base",
    encode_kwargs={"prompt": "passage: ", "normalize_embeddings": True},
    query_encode_kwargs={"prompt": "query: ", "normalize_embeddings": True},
)
```

`embed_documents(list[str])` 批量编码文档，`embed_query(str)` 编码查询。分开是因为 E5 对两类
输入有不同前缀。LangChain 统一了 Embeddings 接口；模型规范仍必须由工程师正确配置。

## 4. `Chroma.add_documents()`

**位置：** `vectorstore.py:add_documents`

```python
vectorstore.add_documents(docs, ids=ids)
```

Chroma 会调用 Embeddings、保存向量、文本和 metadata。V3 仍生成稳定 ID，因为框架不知道
业务上的“同一 Chunk”如何定义。

## 5. `Chroma.as_retriever()`

**位置：** `vectorstore.py:as_retriever`

```python
vectorstore.as_retriever(search_kwargs={"k": top_k})
```

VectorStore 是可写、可搜索的存储；Retriever 是只暴露 `Query -> Documents` 的读取接口。
返回 Retriever 后可以统一使用 `invoke/batch/ainvoke`。V3 也保留这个标准 retriever；需要
distance 的调试链路则使用最小 `ScoredChromaRetriever`。

## 6. `BM25Retriever.from_documents()`

**位置：** `retrieval.py:NativeRetrievalEngine._build_bm25`

```python
BM25Retriever.from_documents(
    self.documents, preprocess_func=chinese_tokenize, k=settings.bm25_k
)
```

BM25 不需要 Embedding 模型，它对分词后的词频、文档频率和长度做统计。中文不能直接照搬
空格分词，因此 V3 注入 jieba tokenizer。框架省去了 BM25 公式和排序实现，但没有替你解决
语言分词。

## 7. `EnsembleRetriever`

**位置：** `retrieval.py:TracedEnsembleRetriever`

```python
TracedEnsembleRetriever(
    retrievers=[vector, bm25], weights=[0.5, 0.5], c=60,
    id_key="document_id",
)
```

它并发/顺序调用多个 Retriever，再以 weighted Reciprocal Rank Fusion 合并名次。`id_key`
让去重基于稳定 Chunk ID，而不是文本。V3 子类只为保留 `fusion_score` 和 route rank；否则
官方实现会返回排序后的 Document，但不会暴露这些调试数据。

## 8. `BaseRetriever.batch()`

**位置：** `retrieval.py:retrieve_queries`

```python
result_sets = self.base_retriever.batch(unique_queries)
```

`invoke(query)` 处理一个 Query；`batch([q1, q2])` 用同一 Runnable 处理多个输入，并保持
结果列表与输入顺序对应。Expansion 在这里真正产生多 Query Retrieval。V3 随后进行二次 RRF，
而不是只做 Union。

## 9. `CrossEncoderReranker.compress_documents()`

**位置：** `reranker.py:ScoredCrossEncoderReranker`

输入是候选 `Sequence[Document]` 和原 Query，输出重新排序并截断的 Document。Embedding 是
“分别编码后比较”；Cross-Encoder 是“Query 和 Document 一起过模型”。它通常更精细但更慢，
所以只处理候选池。

项目没有在最终多 Query 链中使用 `ContextualCompressionRetriever`：该组件会自己先调用一个
单 Query base retriever，而 V3 必须“所有 Query 先融合、再统一 Rerank”。直接调用官方
compressor 更符合算法顺序。

## 10. `ChatPromptTemplate`

**位置：** `query_processing.py` 和 `chain.py`

```python
ANSWER_PROMPT = ChatPromptTemplate.from_messages([...])
```

它把 system/human 消息和 `{question}`、`{context}` 等变量声明成可复用 Runnable。手写版多用
f-string；模板会在 invoke 时检查所需变量，并能直接连接 ChatModel。

## 11. `ChatGoogleGenerativeAI.with_structured_output()`

**位置：** `chain.py:build_rag_chain`

```python
answer_model = model.with_structured_output(AnswerOutput)
```

模型输出被约束并解析成 Pydantic `AnswerOutput`，因此拒答不是靠匹配某一句中文。Expansion 和
Judge 也使用相同思想。Structured output 仍可能调用失败，所以 Query Processing 有 fallback，
Evaluation 会记录错误。

## 12. `RunnableParallel`

**位置：** `chain.py:build_rag_chain`

```python
query_stage = RunnableParallel(
    original_question=RunnablePassthrough(),
    query_processing=processor.as_runnable(),
)
```

同一个输入同时进入两个分支：一边原样保存，另一边 Rewrite/Expansion。返回一个 dict。
这保证最终 Answer 一直使用原问题。它是数据流分叉，不等于 Python 多线程性能承诺。

## 13. `RunnablePassthrough.assign()`

**位置：** `chain.py:build_rag_chain`

```python
RunnablePassthrough.assign(answer_result=answer_chain)
```

它保留 state 中已有字段，并新增 `answer_result`。手写大函数通常不断修改 dict；assign 让
“新增哪个字段”在链定义中可见。

## 14. `RunnableLambda` 与 `RunnableSequence`（`|`）

**位置：** `chain.py:build_rag_chain`

```python
chain = normalize | query_stage | retrieve | rerank | guard | branch
```

`|` 被 LangChain 重载，用来构造 `RunnableSequence`；左侧输出成为右侧输入。`RunnableLambda`
用于把必要的小型 Python 变换（state merge、context formatting）接入统一接口。V3 没有用一个
Lambda 包住完整旧 `ask_rag`，因此 LCEL 确实负责阶段编排。

普通 Python 函数适合纯算法和清晰的数据变换；需要统一 invoke/batch/async、回调、配置或组合时
才值得变成 Runnable。不是函数包装得越多越“原生”。

## 15. `invoke()`、`batch()`、`ainvoke()` 与 `stream()`

- `ask_rag()` 调用 `chain.invoke(question)`：同步单条。
- `batch_ask()` 调用 `chain.batch(questions)`：批量输入。
- `aask_rag()` 调用 `await chain.ainvoke(question)`：异步接口。
- V3 **没有宣称完整结构化响应支持 token streaming**。ChatModel 自身可 stream，但当前链需在
  末尾同时组装 Document、Citation 和 structured refusal；直接 `chain.stream()` 得到的是阶段
  输出，不是已设计好的稳定 SSE 协议。要对外流式输出需另行设计事件格式。

LCEL Pipeline 也不等于 Agent：这里每一步和顺序都由代码固定；Agent 会由模型决定下一工具和路径。

## 暂未使用

- `MultiQueryRetriever`：只提供 Union，不满足本项目的 RRF、Cache 和调试 metadata。
- `ContextualCompressionRetriever`：算法顺序不适合“多 Query 先融合再统一压缩”。
- Callback / LangSmith tracing：框架支持，V3 尚未配置外部 tracing。
- 对外 token streaming：尚未实现稳定 API。

