from langchain.tools import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
from typing import Type
from pydantic.v1 import BaseModel, Field
import logging

from src.pipeline.qa import QAPipeline
from box import Box
import yaml

# --- Tool 1: Semantic Search (our RAG pipeline) ---

class RagToolInput(BaseModel):
    """Input schema for the semantic search tool."""
    query: str = Field(description="The user's question in natural language to search the document database.")

class RagTool(BaseTool):
    """A tool for performing semantic RAG searches."""
    name: str = "semantic_rag_search_tool"
    description: str = (
        "Useful for answering qualitative, open-ended, or 'why' questions about supply chain events "
        "or for retrieving summaries of user behavior. Do not use for questions that require "
        "mathematical calculations, counts, aggregations, or numerical trend analysis."
    )
    args_schema: Type[BaseModel] = RagToolInput

    def _run(self, query: str):
        """Executes the tool."""
        logging.info(f"--- Executing RAG Tool for query: '{query}' ---")
        try:
            with open('config_new.yaml', 'r') as f:
                config = Box(yaml.safe_load(f))
            
            qa_pipeline = QAPipeline(config)
            result = qa_pipeline.run(query)
            return result.get('answer', "Could not generate an answer from the RAG tool.")
        except Exception as e:
            logging.error(f"Error in RagTool: {e}")
            return "An error occurred while processing the semantic search."

    async def _arun(self, query: str):
        """Asynchronous execution is not yet supported."""
        raise NotImplementedError("Asynchronous execution is not available for this tool.")

# --- Tool 2: Pandas DataFrame Agent ---

def get_pandas_agent_tool(config: Box, llm: ChatGoogleGenerativeAI):
    """
    Factory function to create a tool that contains a Pandas Agent.
    This agent can perform quantitative analysis over a DataFrame.
    """
    logging.info("Loading and preparing DataFrame for the Pandas Agent...")
    df = pd.read_csv(config.data.path, encoding='latin1')
    
    # Basic cleaning to make column names Python-friendly
    df.columns = df.columns.str.replace(' ', '_').str.lower()
    df.columns = df.columns.str.replace('[^a-zA-Z0-9_]', '', regex=True)

    pandas_agent_executor = create_pandas_dataframe_agent(
        llm,
        df,
        agent_type="zero-shot-react-description",
        verbose=True,
    )

    class PandasAgentTool(BaseTool):
        name: str = "quantitative_data_analysis_tool"
        description: str = (
            "Indispensable for answering quantitative questions. Use it for any question requiring "
            "counting, summing, averaging, calculating statistics, or analyzing numerical trends over time. "
            "Examples: 'How many orders were there?', 'What is the average price?', 'Show me the evolution of...'"
        )
        args_schema: Type[BaseModel] = RagToolInput

        def _run(self, query: str):
            """Executes the Pandas agent with the given query."""
            logging.info(f"--- Executing Pandas Analysis Tool for query: '{query}' ---")
            try:
                response = pandas_agent_executor.invoke({"input": query})
                return response.get('output', "Could not get an answer from the Pandas agent.")
            except Exception as e:
                logging.error(f"Error in PandasAgentTool: {e}")
                return "An error occurred during data analysis."

        async def _arun(self, query: str):
            raise NotImplementedError("Asynchronous execution is not available for this tool.")

    return PandasAgentTool() 