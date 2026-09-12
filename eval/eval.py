import os
import json
import time

from rag_service import ask_rag

EVAL_REQUEST_INTERVAL_SECONDS = 4

# 1.配置文件路径
TEST_CASE_FILE = "eval/test_case.json"

RESULT_DIR = "eval/results"

RESULT_FILE = os.path.join(RESULT_DIR, "baseline.json")

# 2. 创建results目录
os.makedirs(RESULT_DIR, exist_ok=True)

# 3. 读取测试集
with open(TEST_CASE_FILE, "r", encoding="UTF-8") as file:
    test_cases = json.load(file)

print(f"一共加载{len(test_cases)}个测试用例")

# 4. 初始化统计数据
total_cases = len(test_cases)

answerable_cases = 0
refusal_cases = 0

source_hit_count = 0
page_hit_count = 0
top1_source_hit_count = 0

answer_correct_count = 0
refusal_correct_count = 0

results_output = []

# 5. 逐个执行测试
for case in test_cases:
    print()
    print("=" * 80)
    print(f"测试ID:{case['id']}")
    print(f"类型:{case['type']}")
    print(f"问题:{case['question']}")

    # --------------------------
    # 限制 Gemini 请求频率,每题间隔秒
    # --------------------------
    time.sleep(
        EVAL_REQUEST_INTERVAL_SECONDS
    )
       
    result = ask_rag(case['question'], 3)

    answer = result['answer']
    documents =result['documents']
    metadatas = result['metadatas']
    distances = result['distances']

    # 提取实际检索来源
    retrieved_sources = [
        metadata['source']
        for metadata in metadatas
    ]

    retrieved_pages = [
        metadata['page']
        for metadata in metadatas
    ]

    # TOP 1信息
    top1_source = None
    top1_page = None
    top1_distance = None

    if len(metadatas) > 0:
        top1_source = metadatas[0]['source']
        top1_page = metadatas[0]['page']

    if len(distances) > 0:
        top1_page = distances[0]

    # 6. 判断当前问题是否应该回答
    should_answer = case['should_answer']

    source_hit = None
    page_hit = None
    top1_source_hit = None
    answer_correct = None
    refusal_correct = None

    # 7. 应该回答的问题
    if should_answer:

        answerable_cases += 1   

        expected_source = case['expected_source']
        expected_page = case['expected_page']

        source_hit = expected_source in retrieved_sources

        if source_hit:
            source_hit_count += 1

        if top1_source == expected_source:
            top1_source_hit_count += 1

        # 正确文件 + 正确页码
        page_hit = False

        if expected_page is not None:
            for metadata in metadatas:
                actual_source = metadata["source"]
                actual_page = metadata["page"]

                if (
                    actual_source  == expected_source
                    and
                    actual_page == expected_page
                ):
                    page_hit = True
                    break

        if page_hit:
            page_hit_count += 1

        expected_keywords = case.get(
            "expected_keywords",
            []
        )


        # 多关键词问题
        if expected_keywords:
            answer_correct = all(
                keyword in answer
                for keyword in expected_keywords
            )
        # 普通问题
        else:
            expected_answer = case.get(
                "expected_answer",
                ""
            )
            answer_correct = (
                expected_answer in answer
            )


        if answer_correct:
            answer_correct_count += 1


    # 8. 应该拒答的问题
    else:

        refusal_cases += 1
        refusal_correct = (
            "无法回答"
            in answer
        )

        if refusal_correct:
            refusal_correct_count += 1


# 9. 打印当前测试结果
    print()
    print("【检索结果】")


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
            f"Top {index}"
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


    print()
    print("【模型回答】")

    print(answer)


    print()
    print("【Evaluation】")


    if should_answer:

        print(
            f"Expected Source："
            f"{case.get('expected_source')}"
        )

        print(
            f"Expected Page："
            f"{case.get('expected_page')}"
        )

        print(
            f"Source Hit："
            f"{source_hit}"
        )

        print(
            f"Top1 Source Hit："
            f"{top1_source_hit}"
        )

        print(
            f"Page Hit："
            f"{page_hit}"
        )

        print(
            f"Answer Correct："
            f"{answer_correct}"
        )

    else:

        print(
            f"Refusal Correct："
            f"{refusal_correct}"
        )


    print(
        f"Top1 Source："
        f"{top1_source}"
    )

    print(
        f"Top1 Page："
        f"{top1_page}"
    )

    if top1_distance is not None:

        print(
            f"Top1 Distance："
            f"{top1_distance:.4f}"
        )


# 10. 保存测试结果
    item_result = {

        "id": case["id"],

        "type": case.get(
            "type"
        ),

        "question": case[
            "question"
        ],

        "should_answer": (
            should_answer
        ),

        "expected_source": (
            case.get(
                "expected_source"
            )
        ),

        "expected_page": (
            case.get(
                "expected_page"
            )
        ),

        "expected_answer": (
            case.get(
                "expected_answer"
            )
        ),

        "expected_keywords": (
            case.get(
                "expected_keywords"
            )
        ),

        "actual_answer": answer,

        "top1_source": (
            top1_source
        ),

        "top1_page": (
            top1_page
        ),

        "top1_distance": (
            top1_distance
        ),

        "retrieved_sources": (
            retrieved_sources
        ),

        "retrieved_pages": (
            retrieved_pages
        ),

        "source_hit": (
            source_hit
        ),

        "top1_source_hit": (
            top1_source_hit
        ),

        "page_hit": (
            page_hit
        ),

        "answer_correct": (
            answer_correct
        ),

        "refusal_correct": (
            refusal_correct
        )
    }


    results_output.append(
        item_result
    )



# 11.计算最终指标
source_hit_rate = 0
page_hit_rate = 0
top1_source_hit_rate = 0
answer_accuracy = 0
refusal_accuracy = 0


if answerable_cases > 0:

    source_hit_rate = (
        source_hit_count
        /
        answerable_cases
    )

    page_hit_rate = (
        page_hit_count
        /
        answerable_cases
    )

    top1_source_hit_rate = (
        top1_source_hit_count
        /
        answerable_cases
    )

    answer_accuracy = (
        answer_correct_count
        /
        answerable_cases
    )


if refusal_cases > 0:

    refusal_accuracy = (
        refusal_correct_count
        /
        refusal_cases
    )


# 12. 输出总体结果
print()
print("=" * 80)
print("Evaluation Summary")
print("=" * 80)

print(
    f"Total Cases："
    f"{total_cases}"
)

print(
    f"Answerable Cases："
    f"{answerable_cases}"
)

print(
    f"Refusal Cases："
    f"{refusal_cases}"
)

print()

print(
    f"Source Hit："
    f"{source_hit_count}"
    f"/{answerable_cases}"
    f" = "
    f"{source_hit_rate:.2%}"
)

print(
    f"Top1 Source Hit："
    f"{top1_source_hit_count}"
    f"/{answerable_cases}"
    f" = "
    f"{top1_source_hit_rate:.2%}"
)

print(
    f"Page Hit："
    f"{page_hit_count}"
    f"/{answerable_cases}"
    f" = "
    f"{page_hit_rate:.2%}"
)

print(
    f"Answer Correct："
    f"{answer_correct_count}"
    f"/{answerable_cases}"
    f" = "
    f"{answer_accuracy:.2%}"
)

print(
    f"Correct Refusal："
    f"{refusal_correct_count}"
    f"/{refusal_cases}"
    f" = "
    f"{refusal_accuracy:.2%}"
)


# 13. 保存baseline
final_output = {

    "summary": {

        "total_cases":
            total_cases,

        "answerable_cases":
            answerable_cases,

        "refusal_cases":
            refusal_cases,

        "source_hit_rate":
            source_hit_rate,

        "top1_source_hit_rate":
            top1_source_hit_rate,

        "page_hit_rate":
            page_hit_rate,

        "answer_accuracy":
            answer_accuracy,

        "refusal_accuracy":
            refusal_accuracy
    },

    "details":
        results_output
}


with open(
    RESULT_FILE,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        final_output,
        file,
        ensure_ascii=False,
        indent=2
    )


print()

print(
    f"评估结果已保存："
    f"{RESULT_FILE}"
)