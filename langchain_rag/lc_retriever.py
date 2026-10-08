from langchain_rag.lc_vectorstore import initialize_vectorstore


def as_retriever(top_k: int = 5):
    if top_k < 1:
        raise ValueError("top_k must be at least 1")

    vectorstore = initialize_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )


def retrieve(query: str, top_k: int = 5):
    return as_retriever(top_k=top_k).invoke(query)

