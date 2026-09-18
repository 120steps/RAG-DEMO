import json
import chromadb

from config import TOP_K
from embedding import embed_text


TEST_CASE_FILE = "eval/test_case.json"

CHROMA_DIR = "./chroma_db"

COLLECTION_NAME = "company_knowledge"

# ==============================
# 1. 连接 Chroma
# ==============================

client = chromadb.PersistentClient(
    path=CHROMA_DIR
)

collection = client.get_collection(
    name=COLLECTION_NAME
)


# ==============================
# 2. 读取现有测试集
# ==============================

with open(
    TEST_CASE_FILE,
    "r",
    encoding="utf-8"
) as file:

    test_cases = json.load(file)


# ==============================
# 3. 初始化统计
# ==============================

total = 0

hit_count = 0


# ==============================
# 4. 开始 Retrieval 测试
# ==============================

for case in test_cases:

    # 无答案问题没有 expected source/page
    # 这里先跳过
    if not case.get(
        "should_answer",
        True
    ):
        continue


    expected_source = case.get(
        "expected_source"
    )

    expected_page = case.get(
        "expected_page"
    )


    # 如果测试用例没有 source/page
    # 也不参与这次测试
    if (
        expected_source is None
        or
        expected_page is None
    ):
        continue


    total += 1


    print()
    print("=" * 80)

    print(
        f"测试 ID：{case['id']}"
    )

    print(
        f"问题：{case['question']}"
    )

    print(
        f"预期 Source：{expected_source}"
    )

    print(
        f"预期 Page：{expected_page}"
    )


    # ==============================
    # 5. Question -> Embedding
    # ==============================

    question_embedding = embed_text(
        case["question"]
    )


    # ==============================
    # 6. Chroma Top-K 检索
    # ==============================

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=TOP_K
    )


    documents = results[
        "documents"
    ][0]

    metadatas = results[
        "metadatas"
    ][0]

    distances = results[
        "distances"
    ][0]


    # ==============================
    # 7. 判断 Source + Page
    #    是否出现在 Top-K
    # ==============================

    hit = False

    hit_rank = None


    for index, metadata in enumerate(
        metadatas,
        start=1
    ):

        actual_source = metadata.get(
            "source"
        )

        actual_page = metadata.get(
            "page"
        )


        if (
            actual_source
            == expected_source
            and
            actual_page
            == expected_page
        ):

            hit = True

            hit_rank = index

            break


    # ==============================
    # 8. 统计
    # ==============================

    if hit:

        hit_count += 1


    # ==============================
    # 9. 输出当前结果
    # ==============================

    print()


    if hit:

        print(
            f"✅ Retrieval Hit"
        )

        print(
            f"命中排名：Top {hit_rank}"
        )

    else:

        print(
            "❌ Retrieval Miss"
        )


    print()
    print("召回结果：")


    for index, (
        document,
        metadata,
        distance
    ) in enumerate(
        zip(
            documents,
            metadatas,
            distances
        ),
        start=1
    ):

        print()

        print(
            f"--- Top {index} ---"
        )

        print(
            f"Source："
            f"{metadata.get('source')}"
        )

        print(
            f"Page："
            f"{metadata.get('page')}"
        )

        print(
            f"Chunk："
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"Distance："
            f"{distance:.4f}"
        )

        print(
            f"Content："
            f"{document}"
        )


# ==============================
# 10. 最终统计
# ==============================

print()
print("=" * 80)

print(
    "Retrieval Evaluation Summary"
)

print("=" * 80)


hit_rate = (
    hit_count / total
    if total > 0
    else 0
)


print(
    f"TOP_K：{TOP_K}"
)

print(
    f"Total Cases：{total}"
)

print(
    f"Hit Cases：{hit_count}"
)

print(
    f"Miss Cases：{total - hit_count}"
)

print(
    f"Hit Rate：{hit_rate:.2%}"
)
