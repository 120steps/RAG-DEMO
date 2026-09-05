import pymupdf


pdf_path = "data/pdf/travel_policy.pdf"

doc = pymupdf.open(pdf_path)


print(f"PDF页数：{len(doc)}")


for page_index, page in enumerate(doc):

    text = page.get_text(
        "text",
        sort=True
    )

    print()
    print("=" * 60)

    print(
        f"Page {page_index + 1}"
    )

    print(text)


doc.close()