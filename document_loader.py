import os
import pymupdf

from config import CHUNK_SIZE, CHUNK_OVERLAP, TOP_K

def load_pdf(file_path):
    documents = []
    filename = os.path.basename(file_path)
    doc = pymupdf.open(file_path)
    for page_index, page in enumerate(doc):
        text = page.get_text("text", sort=True).strip()

        if not text:
            continue

        chunks = split_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

        for chunk_id, chunk in enumerate(chunks):
            documents.append({
                "text": chunk,
                "source": filename,
                "page": page_index + 1,
                "chunk_id": chunk_id
            })

    doc.close()
    return documents

def split_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    将文本分割为指定大小的块，并允许重叠。
    :param text: 输入文本
    :param chunk_size: 每个块的最大字符数
    :param overlap: 块之间的重叠字符数
    :return: 文本块列表
    """
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

if __name__ == "__main__":
    pdf_path = "data/pdf/company_policy.pdf"
    documents = load_pdf(pdf_path)
    for doc in documents:
        print(f"Page {doc['page']} from {doc['source']}:")
        print(doc['text'])
        print("=" * 40)