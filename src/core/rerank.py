import logging
import torch
from sentence_transformers import CrossEncoder
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Reranker:
    """A wrapper for a Cross-Encoder model to rerank documents."""
    def __init__(self, model_name: str, device: str = None):
        """
        Initializes the Reranker.

        Args:
            model_name (str): The name of the Cross-Encoder model to load.
            device (str, optional): The device to use ('cuda', 'cpu'). Auto-detected if None.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            logging.info(f"Loading reranker model: {model_name}...")
            self.model = CrossEncoder(model_name, device=self.device)
            logging.info(f"Use pytorch device: {self.device}")
            logging.info("Reranker model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load reranker model {model_name}: {e}")
            raise

    def rerank(self, query: str, documents: List[Dict], top_n: int = 3) -> List[Dict]:
        """
        Reranks a list of documents based on their relevance to a query.

        Args:
            query (str): The search query.
            documents (List[Dict]): A list of documents, where each is a dict with a 'content' key.
            top_n (int): The number of top documents to return.

        Returns:
            List[Dict]: The top_n reranked documents, with a 'rerank_score' added to their metadata.
        """
        if not documents:
            return []
            
        doc_contents = [doc['content'] for doc in documents]
        pairs = [[query, doc_content] for doc_content in doc_contents]
        
        try:
            scores = self.model.predict(pairs, show_progress_bar=False)
            
            # Combine documents with their scores
            for doc, score in zip(documents, scores):
                doc['metadata']['rerank_score'] = score
            
            # Sort documents by the new rerank score in descending order
            reranked_docs = sorted(documents, key=lambda x: x['metadata']['rerank_score'], reverse=True)
            
            return reranked_docs[:top_n]
        except Exception as e:
            logging.error(f"An error occurred during reranking: {e}")
            return documents[:top_n] # Fallback to original order 