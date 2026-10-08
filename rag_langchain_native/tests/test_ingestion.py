from pathlib import Path

from rag_langchain_native.ingestion import load_pdf_pages, split_pages, ingest_pdf


def test_pdf_loading_and_one_based_metadata(v3_settings):
    pdf = v3_settings.project_root / "data" / "pdf" / "travel_policy.pdf"
    pages = load_pdf_pages(pdf)
    assert pages
    assert pages[0].metadata["source"] == "travel_policy.pdf"
    assert pages[0].metadata["page"] == 1

    chunks = split_pages(pages, v3_settings)
    assert chunks
    assert chunks[0].metadata["page"] >= 1
    assert chunks[0].metadata["chunk_id"] == 0
    assert len(chunks[0].metadata["document_id"]) == 24


def test_reingest_replaces_same_source_without_duplicates(
    fake_store, v3_settings
):
    pdf = v3_settings.project_root / "data" / "pdf" / "expense_policy.pdf"
    first = ingest_pdf(pdf, settings=v3_settings, vectorstore=fake_store)
    first_count = fake_store._collection.count()
    second = ingest_pdf(pdf, settings=v3_settings, vectorstore=fake_store)
    assert first["document_ids"] == second["document_ids"]
    assert fake_store._collection.count() == first_count

