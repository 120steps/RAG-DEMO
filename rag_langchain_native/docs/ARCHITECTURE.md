# V3 架构说明

## 1. 设计边界

V3 的 Python 业务代码全部位于 `rag_langchain_native/`。运行时只读根目录的 PDF、`.env`
和测试集，不调用 V1/V2 业务函数；Chroma、上传文件和 Cache 全部写入 V3 `runtime/`。

## 2. 完整执行流程

```mermaid
flowchart TD
    Q[Original Question] --> P[RunnableParallel]
    P --> O[保留 original_question]
    P --> QP[QueryProcessor]
    QP --> RW[Rewrite: Prompt | Gemini | Parser]
    RW --> EX[Expansion: Prompt | Structured Gemini]
    EX --> MQ[BaseRetriever.batch 多 Query]
    MQ --> VR[ScoredChromaRetriever]
    MQ --> BR[BM25Retriever + jieba]
    VR --> ER[TracedEnsembleRetriever / weighted RRF]
    BR --> ER
    ER --> QR[多 Query RRF + stable document_id 去重]
    QR --> CE[CrossEncoderReranker compressor]
    CE --> G[Deterministic answerability guard]
    G -->|拒答| F[结构化 refusal]
    G -->|可答| C[Context construction]
    C --> AP[ChatPromptTemplate]
    AP --> GM[ChatGoogleGenerativeAI structured output]
    GM --> AO[answer + refused + refusal_reason]
    AO --> CIT[Citations from Document.metadata]
    F --> CIT
```

入口 `chain.py:build_rag_chain()` 返回真正的 LCEL `RunnableSequence`。业务调用
`chain.py:ask_rag()` 最终执行 `chain.invoke(question)`；没有把旧版完整函数包在一个
`RunnableLambda` 中。

## 3. 文件与调用关系

| 文件 | 核心职责 | 上游 | 下游 |
|---|---|---|---|
| `config.py` | V3 独立配置、路径和开关 | CLI/API/所有组件 | 无 |
| `ingestion.py` | PDF page Document、Chunk、稳定 ID、重建 | CLI/API upload | `vectorstore.py` |
| `embedding.py` | E5 HuggingFaceEmbeddings | Chroma | Sentence Transformers |
| `vectorstore.py` | V3 Chroma、`as_retriever()` | ingestion/retrieval | langchain-chroma |
| `retrieval.py` | Vector、BM25、RRF、多 Query | chain/eval | Chroma/BM25Retriever |
| `reranker.py` | 本地 Cross-Encoder 文档压缩 | chain | LangChain compressor |
| `query_processing.py` | Rewrite/Expansion LCEL 与独立 Cache | chain | Gemini（开关开启时） |
| `chain.py` | LCEL 总编排、Guard、Prompt、Answer、Citation | CLI/API/eval | 所有 V3 组件 |
| `api.py` | 独立 FastAPI | HTTP | chain/ingestion |
| `cli.py` | 独立命令行 | PowerShell | chain/ingestion |
| `eval/*` | Retrieval、Answer、Judge、消融和版本比较 | CLI | 只写 V3 results |

## 4. Ingestion Pipeline

`PyMuPDFLoader(mode="page")` 将 PDF 每页变成 `Document`。Loader 的 `page` 是从 0
开始，`load_pdf_pages()` 明确加 1；测试集也使用从 1 开始的页码。

`RecursiveCharacterTextSplitter.split_documents()` 在保留页级 metadata 的同时切块。
V3 重新写入 `source/page/chunk_id/document_id`。`document_id` 由
`source|page|chunk_id` 的 SHA-256 前 24 位生成，因此稳定且适合去重。重复入库先仅删除 V3
collection 中同 `source` 的块，然后重建，不会无限累积。

## 5. Retrieval Pipeline

1. `ScoredChromaRetriever` 使用 LangChain Chroma 的
   `similarity_search_with_score()`，把 cosine distance 放进 metadata。
2. `BM25Retriever.from_documents()` 使用 jieba 分词，保留原 `Document`。
3. `TracedEnsembleRetriever` 继承 LangChain `EnsembleRetriever`，沿用 weighted RRF，
   仅扩展 metadata 以保留 `fusion_score`、route rank 和 matched route。
4. Expansion 产生多个 Query 时，`BaseRetriever.batch()` 批量检索，再做 query-level RRF。
   这里没有使用 `MultiQueryRetriever`，因为当前实现只 Union，不能满足 RRF 和调试分数要求。
5. Reranker 在所有 Query 先融合之后统一执行，避免每个 Query 重复 Cross-Encoder 计算。

BM25 擅长词面、编号和专有词精确命中；向量检索擅长同义表达。Hybrid 把两种排序融合，
可能提高召回与首位排序，但不会保证一定提升。

## 6. Reranker

`HuggingFaceCrossEncoder` 同时读取 `(query, document)`，比独立编码再算距离能建模更细的
词间关系。`ScoredCrossEncoderReranker` 只扩展官方 compressor：复制 `Document` 并在
metadata 写入 `rerank_score`。它只能重排候选池；正确 Chunk 没被 Vector/BM25 找到时，
Reranker 无法凭空恢复它。

## 7. Answer、Refusal 与 Citation

Answer Prompt 和 Gemini 使用 structured output `AnswerOutput`，返回 `answer/refused/
refusal_reason`。检索阈值 Guard 可以提前拒答；若检索分数通过但 Context 没有问题所需事实，
Gemini 仍可通过结构化字段拒答。Citation 从最终 `Document.metadata` 生成，LLM 没有生成来源的
权限。

## 8. LangChain 原生组件

- `Document`、`PyMuPDFLoader`、`RecursiveCharacterTextSplitter`
- `HuggingFaceEmbeddings`、`Chroma`、`VectorStoreRetriever`
- `BM25Retriever`、`EnsembleRetriever`
- `HuggingFaceCrossEncoder`、`CrossEncoderReranker`
- `ChatPromptTemplate`、`ChatGoogleGenerativeAI`、structured output
- `RunnableSequence`、`RunnableParallel`、`RunnablePassthrough`、`RunnableLambda`、
  `RunnableBranch`、`invoke/batch/ainvoke`

自定义代码仍负责：稳定 ID、中文 tokenizer、可观测 RRF metadata、多 Query 二次 RRF、
Answer Guard、Citation DTO、Cache 和评估。原因是这些是项目契约，不是框架能自动决定的策略。

## 9. API 与 Evaluation

FastAPI 8011 的 `/upload` 先写 `runtime/uploads/`，然后对同一个 service vectorstore 入库并
刷新 BM25 snapshot，所以上传和检索使用同一知识库。Evaluation 只读根测试集；Retrieval
命中严格按 `expected_source + expected_page`，Answer/Judge 不影响 Retrieval 指标。

