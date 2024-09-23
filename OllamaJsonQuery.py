import re
import json
from langchain.llms import Ollama
from langchain import LLMChain
from langchain.prompts import PromptTemplate

class Keywards:
    def __init__(self, keywords):
        """
        Initialize the Model class with keywords.

        :param keywords: A list of keywords.
        """
        self.keywords = keywords

class OllamaJSONQuery:
    def __init__(self, model_name: str):
        """
        Initialize the OllamaJSONQuery class with the specified model name.
        
        :param model_name: The name of the Ollama model to be used.
        """
        # Initialize the Ollama model
        self.llm = Ollama(model=model_name)
        
        # Define the prompt template to ensure JSON output
        self.prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""
            You are an AI assistant. Find the key words or phrases in the following query and respond in JSON format.
            
            User Query: {query}
            
            The output should be a valid JSON object. Example:
            {{
                "keyworks": ["A", "B"],
            }}
            """
        )
        
        # Create the Langchain LLMChain
        self.llm_chain = LLMChain(
            prompt=self.prompt_template,
            llm=self.llm
        )
    
    def query(self, user_query: str):
        """
        Query the Ollama model and return the response in JSON format.

        :param user_query: The query string to be passed to the model.
        :return: Parsed JSON output or the raw response if invalid JSON.
        """
        # Get the response from the Ollama model
        response = self.llm_chain.run({"query": user_query})
        # extract the json from the response as the response may contain none json content
        json_text = re.search(r'\{.*\}', response, re.DOTALL).group()
        
        # Try to parse the response as JSON
        try:
            json_output = json.loads(json_text)
            return json_output["keywords"]
        except json.JSONDecodeError:
            print("Invalid JSON response. Returning raw text response.")
            return response
