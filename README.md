# 本地 RAG Demo

这是一个用于学习和实践 RAG（Retrieval-Augmented Generation，检索增强生成）的本地 Demo 项目。

项目目标是从零开始搭建一个可以运行的企业知识库问答系统，逐步理解并实现：

* Python 开发环境
* 大模型 API 调用
* 本地 Embedding
* 文档切分 Chunk
* 向量检索
* ChromaDB 本地向量数据库
* RAG 问答流程
* Git / GitHub 代码管理
* 后续扩展 FastAPI、Docker 和云端部署

---

## 1. 项目目标

本项目主要用于学习 RAG 的完整工作流程。

整体流程如下：

```text
本地知识文档
    ↓
Chunk 切分
    ↓
本地 Embedding
    ↓
ChromaDB 向量数据库
    ↓
用户问题
    ↓
问题 Embedding
    ↓
向量相似度检索
    ↓
Top-K 相关知识
    ↓
Context + Question
    ↓
调用大模型
    ↓
生成最终答案
```

RAG 的核心思想不是让大模型直接回答问题，而是：

1. 先从自己的知识库中检索相关内容；
2. 再把检索到的内容作为 Context 提供给大模型；
3. 最后由大模型基于知识库内容生成答案。

---

## 2. 当前技术栈

目前项目主要使用以下技术：

### 开发环境

* Windows
* VS Code
* Python 3.11
* Python Virtual Environment `.venv`

### AI / RAG

* OpenAI API：用于最终大模型问答
* Sentence Transformers：用于本地生成 Embedding
* ChromaDB：本地向量数据库
* NumPy：基础向量计算

### 工程管理

* Git
* GitHub
* `.env`：保存 API Key
* `.gitignore`：避免敏感文件和本地运行数据上传 GitHub

---

## 3. 项目目录

当前项目结构大致如下：

```text
rag-demo/
│
├── data/
│   └── company_policy.txt
│
├── chroma_db/
│
├── ingest.py
├── rag.py
├── simple_rag.py
├── first_ai.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

各文件作用如下。

### `data/`

保存本地知识库文件。

目前使用：

```text
company_policy.txt
```

作为测试企业制度知识库。

---

### `first_ai.py`

第一个大模型 API 测试程序。

主要用于验证：

```text
Python
↓
OpenAI API
↓
大模型
↓
返回回答
```

这一步不包含 RAG。

---

### `simple_rag.py`

最初的手工 RAG Demo。

主要用于理解：

* Chunk
* Embedding
* 相似度计算
* Retrieval
* Context
* LLM

这一版本没有真正的向量数据库，Embedding 主要保存在 Python 内存中。

---

### `ingest.py`

知识库入库程序。

负责：

```text
读取文档
↓
Chunk 切分
↓
本地 Embedding
↓
写入 ChromaDB
```

这个流程也可以称为：

```text
Indexing Pipeline
```

通常只有知识库发生变化时才需要重新执行。

---

### `rag.py`

RAG 查询程序。

负责：

```text
用户输入问题
↓
问题 Embedding
↓
ChromaDB 检索
↓
获取 Top-K Chunk
↓
组成 Context
↓
调用大模型
↓
返回答案
```

这是当前主要的 RAG 查询入口。

---

### `chroma_db/`

ChromaDB 本地向量数据库目录。

用于保存：

* Chunk
* Embedding
* 文档内容
* 后续 Metadata

该目录属于运行时产生的数据，因此不会上传 GitHub。

---

### `.env`

保存本地环境变量，例如：

```env
OPENAI_API_KEY=your_api_key
```

`.env` 不会上传 GitHub。

---

### `.gitignore`

用于忽略不应该提交到 GitHub 的文件。

建议内容：

```gitignore
.venv/
.env
__pycache__/
*.pyc
chroma_db/
```

---

## 4. 环境准备

### 4.1 创建虚拟环境

```bash
python -m venv .venv
```

Windows 激活虚拟环境：

```bash
.venv\Scripts\activate
```

激活成功后终端前面通常会显示：

```text
(.venv)
```

---

## 5. 安装依赖

目前主要依赖包括：

```bash
python -m pip install openai
python -m pip install python-dotenv
python -m pip install sentence-transformers
python -m pip install chromadb
python -m pip install numpy
```

后续可以通过：

```bash
pip freeze > requirements.txt
```

生成依赖文件。

其他人拿到代码后可以执行：

```bash
pip install -r requirements.txt
```

恢复运行环境。

---

## 6. API Key 配置

项目通过 `.env` 保存 API Key。

`.env` 示例：

```env
OPENAI_API_KEY=your_api_key
```

Python 中通过：

```python
from dotenv import load_dotenv

load_dotenv()
```

读取环境变量。

不要把真实 API Key：

* 写死在 Python 代码中；
* 提交到 GitHub；
* 发到聊天记录或截图中。

---

## 7. 知识库入库

首先准备：

```text
data/company_policy.txt
```

然后运行：

```bash
python ingest.py
```

程序会完成：

```text
TXT
↓
Chunk
↓
Embedding
↓
ChromaDB
```

首次运行后，会在本地生成：

```text
chroma_db/
```

---

## 8. 运行 RAG

知识库完成入库后：

```bash
python rag.py
```

然后输入问题，例如：

```text
出差住宿一天最多多少钱？
```

系统会：

1. 对问题生成本地 Embedding；
2. 从 ChromaDB 搜索相关 Chunk；
3. 获取 Top-K 知识；
4. 把知识作为 Context；
5. 调用大模型；
6. 返回最终答案。

---

## 9. 当前已经完成

目前项目已经完成：

* Python 本地开发环境
* VS Code 开发环境
* Python 虚拟环境
* OpenAI API 调用
* `.env` API Key 管理
* 本地知识库 TXT
* Chunk 切分
* 本地 Embedding
* 基础相似度检索
* RAG 基础流程
* ChromaDB 本地向量库接入
* Git / GitHub 代码管理基础

---

## 10. 后续计划

后续计划逐步增加以下能力。

### 第一阶段：完善基础 RAG

* ChromaDB 稳定运行
* Top-K 检索
* Metadata
* Source
* Citation
* Retrieval Threshold

目标：

让回答可以明确告诉用户：

```text
答案：员工需要在出差结束后 30 天内提交报销。

来源：
company_policy.txt
Chunk 3
```

---

### 第二阶段：支持真实文档

支持：

* PDF
* Word
* Markdown
* 多文件知识库

并增加：

* 文件名
* 页码
* 文档类型
* Chunk ID

等 Metadata。

---

### 第三阶段：优化 RAG 效果

逐步学习：

* Chunk Size
* Chunk Overlap
* Top-K
* Metadata Filter
* Query Rewrite
* Hybrid Search
* Reranker
* RAG Evaluation

---

### 第四阶段：服务化

使用 FastAPI 把 RAG 做成 API。

例如：

```text
POST /chat
POST /documents
GET /health
```

使其他 Web 页面或应用可以调用 RAG。

---

### 第五阶段：Docker

使用 Docker 将：

```text
Python
RAG
FastAPI
依赖
```

打包成统一的 Container。

实现：

```text
本地运行
↓
Docker
↓
云服务器部署
```

---

### 第六阶段：云端部署

最终目标架构：

```text
GitHub
   ↓
Cloud Server
   ↓
Docker
   ↓
FastAPI
   ↓
RAG
   ↓
Vector Database
   ↓
LLM
```

未来可以部署到：

* 华为云
* AWS
* Azure
* 阿里云
* 腾讯云
* 其他 Linux 云服务器

---

## 11. 当前项目定位

这个项目当前不是生产级 RAG 系统。

它主要用于：

* 学习 RAG 原理；
* 理解 AI 应用工程结构；
* 学习 Python AI 开发；
* 学习向量数据库；
* 学习大模型 API；
* 学习 Git / GitHub；
* 为后续 FDE / AI 应用开发能力打基础。

项目会按照：

```text
能运行
↓
理解原理
↓
工程化
↓
优化效果
↓
服务化
↓
部署
```

的方式逐步演进，而不是一开始就加入大量复杂框架。

---

## 12. RAG 核心概念

目前这个项目涉及的核心概念包括：

```text
Document
↓
Chunk
↓
Embedding
↓
Vector Database
↓
Similarity Search
↓
Top-K
↓
Context
↓
Prompt
↓
LLM
↓
Answer
```

理解这条链路，是后续学习：

* LangChain
* LlamaIndex
* Reranker
* Hybrid Search
* GraphRAG
* Agentic RAG

的基础。

---

## 13. 项目最终目标

最终希望将该 Demo 逐步发展为一个简单的：

**Enterprise Knowledge Assistant**

即企业知识库助手。

支持：

```text
上传企业文档
↓
自动解析和入库
↓
用户自然语言提问
↓
知识库检索
↓
大模型回答
↓
返回引用来源
```

并最终通过 FastAPI + Docker 部署成为一个可以被其他应用调用的 RAG 服务。
