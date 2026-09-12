from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('intfloat/multilingual-e5-base')

def embed_text(text):
    return embedding_model.encode(text).tolist()

def embed_texts(texts: list[str]):
    return embedding_model.encode(texts).tolist()