from sentence_transformers import SentenceTransformer
import logging
import os
import torch
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmbeddingModel:
    """A wrapper class for a sentence-transformer embedding model."""
    def __init__(self, model_name: str, device: str = None):
        """
        Initializes the embedding model.
        
        Args:
            model_name (str): The name of the model to load from Hugging Face.
            device (str, optional): The device to use ('cuda', 'cpu'). If None, it will be auto-detected.
        """
        logging.info(f"Loading embedding model: {model_name}...")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # BGE models require `trust_remote_code=True`
        is_bge = "bge" in self.model_name.lower()
        self.model = SentenceTransformer(
            self.model_name, 
            device=self.device, 
            trust_remote_code=is_bge
        )

        # Fix the pad_token if it is not set, a common cause of errors.
        if self.model.tokenizer.pad_token is None:
            self.model.tokenizer.pad_token = self.model.tokenizer.eos_token
        
        logging.info(f"Using device: {self.device}")
        logging.info("Embedding model loaded successfully.")

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generates embeddings for a list of texts.
        
        Args:
            texts (List[str]): A list of texts to encode.
        
        Returns:
            List[List[float]]: A list of embeddings.
        """
        logging.info(f"Generating embeddings for {len(texts)} texts...")
        
        # BGE models recommend a prefix for retrieval-focused queries
        if "bge" in self.model_name.lower():
            instruction = "Represent this sentence for searching relevant passages: "
            texts = [instruction + text for text in texts]

        embeddings = self.model.encode(
            texts, 
            show_progress_bar=False # Disabled for a cleaner app log
        )
        logging.info("Embeddings generated successfully.")
        return embeddings.tolist() 