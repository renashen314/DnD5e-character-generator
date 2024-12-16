from sentence_transformers import SentenceTransformer

def get_embedding_function():
    embedding_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    return embedding_model


