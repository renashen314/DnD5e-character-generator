import faiss
import numpy as np
from embedding_mpnet import get_embedding_function
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain



FAISS_PATH = "faiss_index"

PROMPT_TEMPLATE = """
You are tasked with creating a Dungeons & Dragons 5e character sheet. 
Based on the context: ```{context_text}```
Please provide the details in JSON format adhering to the following schema:

{format_instructions}
If the specification existed, use the specification during the task:

specification: ```{specification}```

Provide:
1. A unique character name based on the race.
2. A class from the DnD 5e rules (e.g., Wizard, Fighter, Rogue).
3. Attributes including strength, dexterity, constitution, intelligence, wisdom, and charisma. Use numbers from 1 to 20.
4. Starting equipment based on the class.
5. A short backstory for the character, be creative and provide as many details as possible.

"""

RESPONSE_SCHEMA = [
    ResponseSchema(name="character_name", description="Name of the character"),
    ResponseSchema(name="class", description="The character's class"),
    ResponseSchema(name="attributes", description="The character's attributes like strength, dexterity, etc."),
    ResponseSchema(name="equipment", description="The character's starting equipment"),
    ResponseSchema(name="background", description="The character's backstory"),
]

def get_structured_output(specification, query_text: str = "What are the rules for character creation?"):
    #==================Retrieval===================#
    embedding_function = get_embedding_function()
    # Load the FAISS index 
    index = faiss.read_index(FAISS_PATH + "/faiss_index_with_ids.idx")  # Load your FAISS index
    metadata = np.load(FAISS_PATH + "/metadata.npy", allow_pickle=True)  # Load the metadata
    query_embedding = embedding_function.encode([query_text], convert_to_tensor=False)
    # Perform similarity search in FAISS
    distances, indices = index.search(np.array(query_embedding), k=5)
    context_text = "\n\n---\n\n".join([metadata[i]["page_content"] for i in indices[0]])

    #=============Prompt Structuring===============#
    # Parse the output as JSON
    # prompt_template = PROMPT_TEMPLATE.format(context=context_text, question=query_text)
    output_parser = StructuredOutputParser.from_response_schemas(RESPONSE_SCHEMA)
    format_instructions = output_parser.get_format_instructions()
    # prompt = ChatPromptTemplate.from_template(template=PROMPT_TEMPLATE)
    # chainning
    prompt = ChatPromptTemplate(
        messages=[
            HumanMessagePromptTemplate.from_template(PROMPT_TEMPLATE)
        ],
        specification=["specification"],
        partial_variables = {"format_instructions": format_instructions},
        output_parser=output_parser
    )
    query = prompt.format_messages(specification=specification, format_instructions=format_instructions, context_text=context_text)
    
    model = OllamaLLM(model="llama3.2:1b")
    response_text = model.invoke(query)
    # chain = LLMChain(llm=llm, 
    #                  prompt=prompt)
    # response = chain.predict_and_parse(specification="")
    
    return response_text