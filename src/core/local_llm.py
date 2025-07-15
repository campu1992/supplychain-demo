import logging
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LocalSummarizer:
    """
    A wrapper for a local summarization model using Hugging Face Transformers.
    """
    def __init__(self, model_name: str = 'sshleifer/distilbart-xsum-6-6'):
        """
        Initializes the summarization pipeline.
        The model will be downloaded from Hugging Face hub on first use.

        Args:
            model_name (str): The name of the summarization model to use.
        """
        try:
            logging.info(f"Loading local summarization model: {model_name}...")
            # Using device=-1 forces CPU usage, which is safer for machines without a dedicated GPU.
            self.summarizer = pipeline("summarization", model=model_name, device=-1)
            logging.info("Local summarization model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load local summarization model {model_name}: {e}")
            raise

    def summarize(self, text: str, min_length: int = 20, max_length: int = 100) -> str:
        """
        Generates a summary for a given text.

        Args:
            text (str): The text to summarize.
            min_length (int): The minimum length of the summary.
            max_length (int): The maximum length of the summary.

        Returns:
            str: The generated summary.
        """
        try:
            # The summarization pipeline expects a list of texts.
            # It returns a list of dictionaries.
            summary_result = self.summarizer(
                [text], 
                max_length=max_length, 
                min_length=min_length, 
                do_sample=False,
                truncation=True  # Explicitly enable truncation for long inputs
            )
            return summary_result[0]['summary_text']
        except Exception as e:
            logging.error(f"An error occurred during local summarization: {e}")
            return ""

# Singleton instance for the local summarizer
local_summarizer = LocalSummarizer() 