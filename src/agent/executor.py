import os
from box import Box
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from src.core.tools import RagTool, get_pandas_agent_tool

# Standard ReAct prompt template instructing the agent on how to use tools.
AGENT_PROMPT_TEMPLATE = """
You are an expert supply chain analysis assistant. Answer the following questions as best you can.

You have access to the following tools:

{tools}

Use the following format (the keywords in English are mandatory):

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""

def create_agent_executor():
    """
    Creates and configures the main agent with its tools.
    """
    # 1. Load configuration and LLM
    with open('config_new.yaml', 'r') as f:
        config = Box(yaml.safe_load(f))

    llm = ChatGoogleGenerativeAI(
        model=config.llm.model_name,
        temperature=0,  # Agent needs to be as deterministic as possible
        google_api_key=os.getenv("GEMINI_API_KEY", config.llm.api_key),
    )

    # 2. Initialize tools
    tools = [
        RagTool(),
        get_pandas_agent_tool(config, llm)
    ]
    
    # 3. Create the agent prompt
    prompt = PromptTemplate.from_template(AGENT_PROMPT_TEMPLATE)

    # 4. Create the ReAct agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create and return the AgentExecutor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # To see the agent's reasoning in the console
        handle_parsing_errors=True,
        max_iterations=5 # Avoid infinite loops
    )

    return agent_executor 