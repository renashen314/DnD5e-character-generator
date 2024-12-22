import faiss
import numpy as np
from embedding_mpnet import get_embedding_function
from load_pdf import load_documents, split_documents
import argparse
import os
import gc
import shutil

FAISS_PATH = "faiss_index"

def create_faiss_index(chunks: list[str]):
    try:
        cleanup_resources(embeddings, chunks)
        print("👉 Generating embeddings...")
        embedding_function = get_embedding_function()
        embeddings = embedding_function.encode(chunks, batch_size=16, convert_to_tensor=False)


        # Create a FAISS index (FlatL2 is a basic index type)
        index = faiss.IndexFlatL2(len(chunks))

        # Add embeddings to the index
        print(f"👉 Adding {len(chunks)} chunks to the FAISS index...")
        index.add(np.array(embeddings))
        faiss.write_index(index, FAISS_PATH)

        # metadata = {{"id": f"{chunk.id}", "source":"data/DnD_5e_Players_Handbook.pdf", "page":f"{chunk.metadata.page}", "page_content":chunk } for i, chunk in enumerate(chunks)}
        # np.save("metadata.npy", metadata)
        print(f"✅ Successfully added {len(chunks)} chunks to the index.")
        print(f"✅ Total chunks in database: {index.ntotal}")
        cleanup_resources(embeddings, chunks)
        
        return True
    except Exception as e:
        print(f"Failed to index chunks. Error: {e}")
        cleanup_resources(embeddings, chunks)
        return False
    
def clear_database():
    if os.path.exists(FAISS_PATH):
        shutil.rmtree(FAISS_PATH)

def cleanup_resources(*objects):
    """
    Deletes provided objects and triggers garbage collection.
    """
    for obj in objects:
        del obj  # Delete objects to free memory
    gc.collect()  # Force garbage collection

def main():
    print("start")
    documents = load_documents("data/DnD_5e_Players_Handbook.pdf", page_limit=160)
    chunks = split_documents(documents)

    # cache chunks function:
    # if chunks:
    #     cache_chunks(chunks, cache_path=CACHE_CHUNK)
    # print(chunks[100])
    success = create_faiss_index([chunk.page_content for chunk in chunks])
    if success:
        print("Indexing completed successfully.")
    else:
        print("Indexing process encountered issues.")


if __name__ == "__main__":
    main()