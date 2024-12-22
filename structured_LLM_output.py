import faiss
import numpy as np
from embedding_mpnet import get_embedding_function
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are tasked with creating a Dungeons & Dragons 5e character sheet. 
Please provide the details in JSON format adhering to the following schema:

{format_instructions}

Provide:
1. A unique character name based on the race.
2. A class from the DnD 5e rules (e.g., Wizard, Fighter, Rogue).
3. Attributes including strength, dexterity, constitution, intelligence, wisdom, and charisma. Use numbers from 1 to 20.
4. Starting equipment based on the class.
5. A short backstory for the character.

Be creative and provide a detailed response.
"""

RESPONSE_SCHEMA = [
    ResponseSchema(name="character_name", description="Name of the character"),
    ResponseSchema(name="class", description="The character's class"),
    ResponseSchema(name="attributes", description="The character's attributes like strength, dexterity, etc."),
    ResponseSchema(name="equipment", description="The character's starting equipment"),
    ResponseSchema(name="background", description="The character's backstory"),
]

parser = StructuredOutputParser.from_response_schemas(RESPONSE_SCHEMA)

def get_structured_output(query_text: str = "What are the rules for character creation?"):
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
    format_instructions = parser.get_format_instructions()
    
    prompt = PROMPT_TEMPLATE(input_variables=[], template=PROMPT_TEMPLATE, partial_variables={"format_instructions": format_instructions})
    
    model = OllamaLLM(model="llama3.2:1b")
    response_text = model.invoke(prompt_template)

    # Retrieve source metadata (e.g., IDs)
    sources = [metadata[idx].get("id", None) for idx in indices[0]]

    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    
    return response_text