import faiss
import numpy as np
from embedding_mpnet import get_embedding_function
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate

FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
    Answer the question based only on the following context:

    {context}

    ---

    Answer the question based on the above context: {question}
    """

def create_faiss_index(chunks):
    embedding_function = get_embedding_function()
    embeddings = embedding_function.encode(chunks.page_content, convert_to_tensor=False)
    # Define the dimension of your embeddings
    embedding_dim = embeddings[0].shape[0]

    # Create a FAISS index (FlatL2 is a basic index type)
    index = faiss.IndexFlatL2(embedding_dim)

    # Add embeddings to the index
    index.add(np.array(embeddings))
    faiss.write_index(index, FAISS_PATH)

    metadata = {{"id": f"{chunk.id}", "source":"data/DnD_5e_Players_Handbook.pdf", "page":f"{chunk.metadata.page}", "page_content":chunk } for i, chunk in enumerate(chunks)}
    np.save("metadata.npy", metadata)
    

def query_database(query_text: str = "What are the rules for character creation?"):
    embedding_function = get_embedding_function()
    
    # Load the FAISS index 
    index = faiss.read_index(FAISS_PATH + "/faiss_index_with_ids.idx")  # Load your FAISS index
    metadata = np.load(FAISS_PATH + "/metadata.npy", allow_pickle=True)  # Load the metadata

    # Step 3: Generate the embedding for the query
    query_embedding = embedding_function.encode([query_text], convert_to_tensor=False)

    # Perform similarity search in FAISS
    distances, indices = index.search(np.array(query_embedding), k=5)

    context_text = "\n\n---\n\n".join([metadata[i]["page_content"] for i in indices[0]])
   
    prompt_template = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    
    model = OllamaLLM(model="llama3.2:1b")
    response_text = model.invoke(prompt_template)

    # Retrieve source metadata (e.g., IDs)
    sources = [metadata[idx].get("id", None) for idx in indices[0]]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text
