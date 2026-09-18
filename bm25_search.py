import math
import re
from collections import Counter


ALPHANUMERIC_OR_CJK = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text):
    tokens = []
    for segment in ALPHANUMERIC_OR_CJK.findall(text.lower()):
        if "\u4e00" <= segment[0] <= "\u9fff":
            characters = list(segment)
            tokens.extend(characters)
            tokens.extend(
                "".join(characters[index:index + 2])
                for index in range(len(characters) - 1)
            )
        else:
            tokens.append(segment)
    return tokens


def candidate_key(candidate):
    metadata = candidate["metadata"]
    return (
        metadata.get("source"),
        metadata.get("page"),
        metadata.get("chunk_id"),
    )


class BM25Index:
    def __init__(self, documents, metadatas, k1=1.5, b=0.75):
        if len(documents) != len(metadatas):
            raise ValueError(
                "documents and metadatas must have equal lengths"
            )

        self.documents = documents
        self.metadatas = metadatas
        self.k1 = k1
        self.b = b
        self.tokenized_documents = [
            tokenize(document)
            for document in documents
        ]
        self.term_frequencies = [
            Counter(tokens)
            for tokens in self.tokenized_documents
        ]
        self.document_lengths = [
            len(tokens)
            for tokens in self.tokenized_documents
        ]
        self.document_count = len(documents)
        self.average_document_length = (
            sum(self.document_lengths) / self.document_count
            if self.document_count
            else 0
        )

        document_frequencies = Counter()
        for tokens in self.tokenized_documents:
            document_frequencies.update(set(tokens))

        self.inverse_document_frequencies = {
            token: math.log(
                1
                + (
                    self.document_count
                    - frequency
                    + 0.5
                )
                / (frequency + 0.5)
            )
            for token, frequency in document_frequencies.items()
        }

    @classmethod
    def from_chroma(cls, collection):
        results = collection.get(
            include=["documents", "metadatas"],
        )
        return cls(
            results["documents"],
            results["metadatas"],
        )

    def search(self, query, top_k=10):
        query_tokens = tokenize(query)
        scored_documents = []

        for index, term_frequency in enumerate(self.term_frequencies):
            document_length = self.document_lengths[index]
            score = 0.0

            for token in query_tokens:
                frequency = term_frequency.get(token, 0)
                if not frequency:
                    continue

                idf = self.inverse_document_frequencies.get(token, 0)
                length_normalization = 1 - self.b
                if self.average_document_length:
                    length_normalization += (
                        self.b
                        * document_length
                        / self.average_document_length
                    )
                denominator = (
                    frequency
                    + self.k1 * length_normalization
                )
                score += (
                    idf
                    * frequency
                    * (self.k1 + 1)
                    / denominator
                )

            if score > 0:
                scored_documents.append((score, index))

        scored_documents.sort(reverse=True)
        return [
            {
                "document": self.documents[index],
                "metadata": self.metadatas[index],
                "original_distance": None,
                "bm25_score": float(score),
                "rerank_score": None,
            }
            for score, index in scored_documents[:top_k]
        ]


def merge_candidates(vector_candidates, bm25_candidates):
    merged_by_key = {}
    ordered_keys = []

    for candidate in [*vector_candidates, *bm25_candidates]:
        key = candidate_key(candidate)
        if key not in merged_by_key:
            merged_by_key[key] = {**candidate}
            ordered_keys.append(key)
            continue

        existing = merged_by_key[key]
        if candidate.get("original_distance") is not None:
            existing["original_distance"] = candidate[
                "original_distance"
            ]
        if candidate.get("bm25_score") is not None:
            existing["bm25_score"] = candidate["bm25_score"]

    return [merged_by_key[key] for key in ordered_keys]
