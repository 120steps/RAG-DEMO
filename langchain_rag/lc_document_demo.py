import json

from langchain_core.documents import Document


def main():
    document = Document(
        page_content="国内常规出差应至少提前2个工作日提交申请。",
        metadata={
            "source": "travel_policy.pdf",
            "page": 2,
            "chunk_id": 0,
        },
    )

    print("完整 Document：")
    print(
        json.dumps(
            document.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n读取 page_content：")
    print(document.page_content)

    print("\n读取 metadata：")
    print(document.metadata)

    print("\n读取单个 metadata 字段：")
    print(f"source={document.metadata['source']}")
    print(f"page={document.metadata['page']}")
    print(f"chunk_id={document.metadata['chunk_id']}")


if __name__ == "__main__":
    main()

