import logging
from typing import Dict, List
from box import Box

from src.core.embedding import EmbeddingModel
from src.core.retrieval import ChromaRetriever
from src.core.llm import Gemini
from src.core.prompts import QA_PROMPT
from src.core.rerank import Reranker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class QAPipeline:
    """A pipeline for retrieving context and generating answers."""
    def __init__(self, config: Box):
        self.config = config
        self.embedding_model = EmbeddingModel(
            model_name=config.llm.embedding_model,
            device=config.llm.device
        )
        self.retriever = ChromaRetriever(
            host=config.database.host,
            port=config.database.port,
            collection_name=config.database.collection_name
        )
        self.llm = Gemini(config)
        self.reranker = Reranker(
            model_name=config.llm.reranker_model,
            device=config.llm.device
        )

    def _get_relevant_documents(self, query: str) -> List[Dict]:
        """
        Retrieves and reranks documents for a given query.
        """
        logging.info(f"Retrieving documents for query: '{query}'")
        query_embedding = self.embedding_model.get_embeddings([query])[0]
        
        results = self.retriever.query(
            query_embedding=query_embedding, 
            n_results=self.config.retrieval.k_retrieved_docs
        )
        
        documents = []
        if results and results.get('documents'):
            for i, doc_content in enumerate(results['documents'][0]):
                documents.append({
                    "content": doc_content,
                    "metadata": results['metadatas'][0][i],
                })
        
        logging.info(f"Reranking {len(documents)} documents...")
        reranked_docs = self.reranker.rerank(
            query, 
            documents, 
            top_n=self.config.retrieval.k_reranked_docs
        )
        return reranked_docs

    def _answer_with_context(self, query: str, context_docs: List[Dict]) -> str:
        """
        Uses the LLM to answer a query based on the provided context.
        """
        context_str = "\n\n---\n\n".join([doc['content'] for doc in context_docs])
        prompt = QA_PROMPT.format(query=query, context=context_str)
        
        logging.info("Generating answer with context...")
        answer = self.llm.generate(prompt)
        return answer

    def run(self, query: str) -> Dict:
        """
        Runs the full question-answering pipeline.
        """
        logging.info(f"--- Running QA pipeline for query: '{query}' ---")
        
        relevant_docs = self._get_relevant_documents(query)
        answer = self._answer_with_context(query, relevant_docs)
        
        logging.info("--- QA pipeline completed. ---")
        
        return {
            "query": query,
            "answer": answer,
            "source_documents": relevant_docs
        } 