import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from load_pdf import calculate_chunk_ids


def get_embedding_function():
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")
    return embedding_model


def create_vector_store(chunks: Document):
    embedding_function = get_embedding_function()

    # initialize vector store
    single_vector = embedding_function.embed_query("this is a cat")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embedding_function,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},  # tag chunk id later
    )

    # generate unique ids for chunks
    ids = calculate_chunk_ids(chunks)  # returns an array
    items = vector_store.add_documents(documents=chunks, ids=ids)

    return vector_store


def save_to_local(vector_store, db: str):
    vector_store.save_local(db)
    return True


def load_local_db(db: str, embeddings):
    vector_store = FAISS.load_local(
        db, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


def retrieve_vector_store(vector_store, query):
    retriever = vector_store.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "fetch_k": 100, "lambda_mult": 1}
    )
    docs = retriever.invoke(query)
    for doc in docs:
        print(doc.page_content)
        print("\n----------\n")
    return docs
