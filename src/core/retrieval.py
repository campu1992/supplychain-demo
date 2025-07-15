import chromadb
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChromaRetriever:
    """A retriever class for interacting with a ChromaDB collection."""
    def __init__(self, host: str, port: int, collection_name: str):
        """
        Initializes the ChromaDB client and collection.

        Args:
            host (str): The hostname of the ChromaDB server.
            port (int): The port of the ChromaDB server.
            collection_name (str): The name of the collection to use.
        """
        try:
            self.client = chromadb.HttpClient(host=host, port=port)
            self.collection = self.client.get_or_create_collection(name=collection_name)
            logging.info(f"Connected to ChromaDB at {host}:{port}. Collection '{collection_name}' is ready.")
        except Exception as e:
            logging.error(f"Failed to connect to ChromaDB: {e}")
            raise

    def add_documents(self, documents: List[str], metadatas: List[Dict], ids: List[str], embeddings: List[List[float]]):
        """
        Adds documents to the ChromaDB collection in batches.

        Args:
            documents (List[str]): The document contents.
            metadatas (List[Dict]): The metadata for each document.
            ids (List[str]): The unique IDs for each document.
            embeddings (List[List[float]]): The embeddings for each document.
        """
        logging.info(f"Adding {len(documents)} documents to the vector store...")
        try:
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            logging.info("Documents added successfully.")
        except Exception as e:
            logging.error(f"An error occurred while adding documents: {e}")
            raise

    def query(self, query_embedding: List[float], n_results: int = 10) -> Optional[Dict]:
        """
        Queries the collection for similar documents.

        Args:
            query_embedding (List[float]): The embedding of the query text.
            n_results (int): The number of results to return.

        Returns:
            Optional[Dict]: A dictionary of query results, or None if an error occurs.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )
            return results
        except Exception as e:
            logging.error(f"An error occurred during query: {e}")
            return None

    def clear_collection(self):
        """Clears and recreates the collection to ensure a fresh start."""
        logging.info(f"Attempting to clear collection '{self.collection.name}'...")
        try:
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.get_or_create_collection(name=self.collection.name)
            logging.info("Collection cleared and recreated successfully.")
        except Exception as e:
            logging.error(f"Failed to clear collection: {e}")
            # If clearing fails, it might not exist, so we try to create it as a fallback.
            self.collection = self.client.get_or_create_collection(name=self.collection.name) 