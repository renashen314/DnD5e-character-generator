from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
import json
from transformers import AutoTokenizer, pipeline
from embedding_mpnet import get_embedding_function

CACHE_CHUNK = "chunks.json"

def load_documents(path, page_limit=160):
    loader = loader = PyPDFLoader(path)
    document = loader.load_and_split()
    return document

def split_documents(document, chunk_size=600, chunk_overlap=80):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
        chunks = text_splitter.split_documents(document)
        return chunks
    except Exception as e:
        print(f"Error during document splitting: {e}")
        return None

def calculate_chunk_ids(chunks):

    # This will create IDs like "data/DnD5e.pdf:6:2"
    # Page Source : Page Number : Chunk Index
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks

def cache_chunks(chunks, cache_path=CACHE_CHUNK):
    try:
        serializable_chunks = [chunk.dict() if hasattr(chunk, "dict") else chunk for chunk in chunks]
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(serializable_chunks, f, ensure_ascii=False, indent=4)
        print(f"Chunks successfully cached to {cache_path}.")
    except Exception as e:
        print(f"Error caching chunks: {e}")

def load_cached_chunks(cache_path=CACHE_CHUNK):
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} chunks from cache.")
        return chunks
    except Exception as e:
        print(f"Error loading cached chunks: {e}")
        return None



documents = load_documents("data/DnD_5e_Players_Handbook.pdf")
chunks = split_documents(documents)
# if chunks:
#     cache_chunks(chunks, cache_path=CACHE_CHUNK)
# print(chunks[100])

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")  # Replace with your model
# chunk_texts = load_cached_chunks()
for i, chunk in enumerate(chunks[:5]):  # Analyze first 5 chunks
    tokens = len(tokenizer.encode(chunk.page_content))
    print(f"Chunk {i+1} has {tokens} tokens")

embeddings = get_embedding_function().encode(chunks, convert_to_tensor=False)