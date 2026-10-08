from langchain_core.embeddings import Embeddings

from embedding import embed_text, embed_texts


class ExistingEmbeddingAdapter(Embeddings):
    """Expose the existing project embedding functions to LangChain.

    The handwritten implementation sends raw text directly to
    SentenceTransformer.encode(): it adds no ``query:`` or ``passage:``
    prefix and does not override ``normalize_embeddings``. Delegating to the
    same functions preserves that behavior exactly.
    """

    def embed_query(self, text: str) -> list[float]:
        return embed_text(text)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return embed_texts(texts)

