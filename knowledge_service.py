import chromadb

from document_loader import load_pdf
from embedding import embed_texts

CHROMA_DIR = "./chroma_db"
COLLECTION_NAME = "company_knowledge"

client = chromadb.PersistentClient(path=CHROMA_DIR)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

def add_pdf_to_knowledge(
    pdf_path: str
):
    # 1. pdf -> Page -> chunk
    docs = load_pdf(pdf_path)

    if not docs:
        raise ValueError(f"PDF文件 {pdf_path} 中没有可用的文本内容。")

    texts = [doc["text"] for doc in docs]

    # 2. 本地embedding
    embeddings = embed_texts(
        texts
    )

    ids = []
    metadatas = []

    # 3. 构建ID和metadata
    for doc in docs:
        document_id = (
            f"{doc['source']}"
            f"_page_{doc['page']}"
            f"_chunk_{doc['chunk_id']}"
        )

        ids.append(document_id)

        metadatas.append({
            "source": doc['source'],
            "page": doc['page'],
            "chunkId": doc['chunk_id']
        })

    # 4. 写入chromadb
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    # 5. 返回结果
    return {
        "filename": docs[0]['source'],
        "chunks": len(texts),
        "status": "indexed"
    }