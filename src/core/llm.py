import os
import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from box import Box

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Gemini:
    """A wrapper class for the Google Gemini model."""
    def __init__(self, config: Box):
        """
        Initializes the Gemini model.

        Args:
            config (Box): The application configuration object containing model details.
        """
        self.model_name = config.llm.model_name
        self.temperature = config.llm.temperature
        
        try:
            # Prioritize env var, but fall back to config file
            self.api_key = os.getenv("GEMINI_API_KEY", config.llm.api_key)
            if not self.api_key:
                logging.warning("GEMINI_API_KEY not found in environment or config file. Check your configuration.")
            
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=self.temperature,
                google_api_key=self.api_key,
                convert_system_message_to_human=True
            )
            logging.info("Gemini API configured successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize Gemini API: {e}")
            raise

    def generate(self, prompt: str) -> str:
        """
        Generates a response from the Gemini model.

        Args:
            prompt (str): The prompt to send to the model.

        Returns:
            str: The generated text content.
        """
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            logging.error(f"An error occurred during Gemini API call: {e}")
            return "Error: Could not get a response from the language model." 