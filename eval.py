import json

from rag_service import ask_rag

with open("eval/test_case.json", "r", encoding="utf-8") as f:
    test_cases = json.load(f)

total = len(test_cases)

source_hit_count = 0;
answer_correct_count = 0;
refusal_correct_count = 0;

for case in test_cases:
    print()
    print("=" * 60)

    print(f"测试：{case['id']}")

    print(f"问题：{case['question']}")

    result = ask_rag(case["question"])

    answer = result["answer"]

    metadatas = result["metadatas"]

    retrieved_sources = [
        metadata["source"] 
        for metadata in metadatas
    ]

    print(
        f"期望来源:"
        f"{case['expected_source']}"
    )

    print(
        f"实际来源:"
        f"{retrieved_sources}"
    )

    print(
        f"期望答案:"
        f"{case['expected_answer']}"
    )

    print(
        f"实际答案:"
        f"{answer}"
    )


    # 应该回答的问题
    if case["should_answer"]:
        source_hit = (
            case["expected_source"] in retrieved_sources
        )

        answer_correct = (
            case["expected_answer"] in answer
        )

        if source_hit:
            print("✅ 来源命中")
            source_hit_count += 1

        if answer_correct:
            print("✅ 回答正确")
            answer_correct_count += 1

        print(
            f"Source hit: {source_hit}, \n"
            f"Answer correct: {answer_correct}"
        )

    # 不应该回答的问题
    else:
        refusal_correct = (
            "无法回答" in answer
        )

        if refusal_correct:
            print("✅ 拒绝回答正确")
            refusal_correct_count += 1

        print(
            f"Refusal correct: {refusal_correct}"
        )

print()
print("=" * 60)
print("Evaluation Summary:")
print(f"Total tests: {total}")
print(f"Source hits: {source_hit_count}")
print(f"Answer correct: {answer_correct_count}")
print(f"Refusal correct: {refusal_correct_count}")