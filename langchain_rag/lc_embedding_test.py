import math

from embedding import embed_text, embed_texts
from langchain_rag.lc_embedding import ExistingEmbeddingAdapter


COSINE_TOLERANCE = 1e-9
MAX_ABSOLUTE_DIFFERENCE_TOLERANCE = 1e-7


def cosine_similarity(first, second):
    dot_product = sum(
        left * right
        for left, right in zip(first, second)
    )
    first_norm = math.sqrt(sum(value * value for value in first))
    second_norm = math.sqrt(sum(value * value for value in second))
    if first_norm == 0 or second_norm == 0:
        return 0.0
    return dot_product / (first_norm * second_norm)


def compare_vectors(label, original_vector, langchain_vector):
    same_dimension = len(original_vector) == len(langchain_vector)
    if same_dimension:
        differences = [
            abs(original - adapted)
            for original, adapted in zip(
                original_vector,
                langchain_vector,
            )
        ]
        cosine = cosine_similarity(
            original_vector,
            langchain_vector,
        )
        max_difference = max(differences, default=0.0)
        mean_difference = (
            sum(differences) / len(differences)
            if differences
            else 0.0
        )
    else:
        cosine = None
        max_difference = None
        mean_difference = None

    passed = (
        same_dimension
        and cosine is not None
        and abs(1.0 - cosine) <= COSINE_TOLERANCE
        and max_difference is not None
        and max_difference <= MAX_ABSOLUTE_DIFFERENCE_TOLERANCE
    )

    print(label)
    print("-" * 60)
    print(f"原方法向量维度: {len(original_vector)}")
    print(f"LangChain 向量维度: {len(langchain_vector)}")
    print(f"维度一致: {same_dimension}")
    print(
        "余弦相似度: "
        f"{cosine:.12f}" if cosine is not None else "余弦相似度: N/A"
    )
    print(
        "最大绝对数值差异: "
        f"{max_difference:.12e}"
        if max_difference is not None
        else "最大绝对数值差异: N/A"
    )
    print(
        "平均绝对数值差异: "
        f"{mean_difference:.12e}"
        if mean_difference is not None
        else "平均绝对数值差异: N/A"
    )
    print(f"一致性测试: {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    embeddings = ExistingEmbeddingAdapter()

    query = "国际出差需要提前多久申请？"
    original_query_vector = embed_text(query)
    langchain_query_vector = embeddings.embed_query(query)
    query_passed = compare_vectors(
        "Query Embedding 对比",
        original_query_vector,
        langchain_query_vector,
    )

    document = "国际及港澳台出差至少提前10个工作日提交申请。"
    original_document_vector = embed_texts([document])[0]
    langchain_document_vector = embeddings.embed_documents(
        [document]
    )[0]
    document_passed = compare_vectors(
        "Document Embedding 对比",
        original_document_vector,
        langchain_document_vector,
    )

    if not query_passed or not document_passed:
        raise AssertionError(
            "Embedding 一致性测试失败；不能复用当前 Chroma 向量。"
        )

    print("最终结果: Query 与 Document Embedding 均保持一致。")


if __name__ == "__main__":
    main()
