# Smart Supply Chain Analysis Agent

## 1. Project Overview

This project implements an advanced AI agent for analyzing supply chain data. The system is built around a Retrieval-Augmented Generation (RAG) architecture, enhanced with agentic capabilities using LangChain. It allows users to ask complex questions in natural language and receive data-grounded answers by intelligently choosing between two distinct tools:

*   **Semantic Search (RAG):** For qualitative, open-ended questions (e.g., "why" or "how" something happened).
*   **Quantitative Analysis:** For questions requiring numerical computation, aggregation, and trend analysis (e.g., "how many," "what is the average").

The application is served through a user-friendly Streamlit interface and is fully containerized with Docker for easy deployment.

## 2. Architecture and Design

The system's architecture evolved from a simple RAG pipeline to a sophisticated, multi-tool AI agent to handle a wider variety of user queries.

### Final Architecture: The AI Agent

The core of the project is an **Agent Executor** built with LangChain. This agent acts as a smart router or a "reasoning engine." When it receives a user's question, it doesn't just search for information; it first *thinks* about the best way to answer the question and then selects the appropriate tool for the job.

The agent has access to two specialized tools:

1.  **`semantic_rag_search_tool`**:
    *   **Purpose**: Handles qualitative and context-based questions.
    *   **Mechanism**: This is the classic RAG pipeline. It takes the user's query, embeds it, retrieves relevant document chunks from the ChromaDB vector store, uses a Cross-Encoder to rerank the results for precision, and finally passes the best context to a Gemini LLM to synthesize a final answer.
    *   **Best for**: "Why did late deliveries spike?", "What were the common issues for returns?"

2.  **`quantitative_data_analysis_tool`**:
    *   **Purpose**: Handles quantitative and analytical questions that require exact calculations.
    *   **Mechanism**: This tool is a powerful `Pandas DataFrame Agent`. The agent is given access to the raw `DataCoSupplyChainDataset.csv` loaded into a Pandas DataFrame. When invoked, the LLM writes and executes Python code (using Pandas) on the fly to compute the answer directly from the source data.
    *   **Best for**: "How many orders were there for the Apparel category in May 2018?", "What was the average product price in the Sports category?"

This dual-tool approach gives the system the flexibility to provide both nuanced, context-aware narratives and precise, calculated numerical answers.

## 3. Models and Core Technologies

### Models Used

*   **Embedding Model (`BAAI/bge-large-en-v1.5`):** This state-of-the-art model converts all documents and user queries into high-dimensional vectors (embeddings), crucial for high-quality retrieval.
*   **Reranking Model (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** This Cross-Encoder refines search results by directly comparing the query against retrieved documents, significantly improving context relevance.
*   **Core LLM (`gemini-1.5-flash`):** This powerful Google model acts as the agent's "brain" for tool selection and as the response synthesizer for the RAG pipeline.

### Key Libraries and Frameworks

*   **`LangChain`**: The core framework used to build the agent, create the toolset, and manage the interaction between the LLM and the tools.
*   **`Streamlit`**: Provides the simple and interactive web-based user interface for the application.
*   **`Pandas`**: Used for loading and manipulating the supply chain dataset, forming the backbone of the quantitative analysis tool.
*   **`ChromaDB`**: The vector database that stores all the document embeddings for fast and efficient semantic retrieval.
*   **`Sentence-Transformers`**: The library used to load and run the open-source embedding and reranking models.
*   **`PyTorch`**: The deep learning backend required by the Sentence-Transformers library.
*   **`py-YAML` & `python-box`**: Used for loading and managing the project's configuration from the `config_new.yaml` file.

## 4. How to Run with Docker (Recommended)

This project is designed to be run easily using Docker. The following steps will guide you through the process from setup to launch.

### Prerequisites

*   **Docker Desktop**: Ensure Docker is installed and running on your system.
*   **Git**: For cloning the repository.
*   **A Google Gemini API Key**: To enable the AI agent's reasoning capabilities.

### Step 1: Clone the Repository

Open your terminal (PowerShell, Command Prompt, or bash) and clone the project to your local machine:

```bash
git clone <repository-url>
cd <project-directory>
```

### Step 2: Configure Your API Key

1.  Open the `config_new.yaml` file in the project's root directory.
2.  Find the `api_key` field and replace the placeholder value with your actual Google Gemini API key.

```yaml
# config_new.yaml

llm:
  api_key: "YOUR_REAL_GEMINI_API_KEY" # <-- PASTE YOUR KEY HERE
  # ... other settings
```

### Step 3: Build the Docker Image

This command packages the application and all its dependencies into a Docker image. You only need to run this once, or again if you change the code or `requirements.txt`.

```powershell
docker build -t supply-chain-agent .
```

### Step 4: Build the Vector Database Index

This is a one-time setup step. This command runs a temporary container to process the source data, generate embeddings, and save the resulting vector database into a `chroma_data` folder on your local machine.

```powershell
docker run --rm -v "${PWD}/chroma_data:/app/chroma_data" supply-chain-agent python scripts/build_index.py
```
*(This process may take a few minutes as it downloads models and processes data).*

### Step 5: Launch the Application

This final command starts the application. It runs the container in the background, connects it to the database you just created, and automatically opens the application in your web browser.

```powershell
docker run -d --rm --name supply-chain-app -p 8501:8501 -v "${PWD}/chroma_data:/app/chroma_data" supply-chain-agent; Start-Sleep -Seconds 5; Start-Process http://localhost:8501
```

The application will be running at `http://localhost:8501`.

### Managing the Application

Since the container runs in the background, use these commands to manage it:

*   **View Logs**: To see the agent's "thoughts" and other logs from the application, run:
    ```powershell
    docker logs -f supply-chain-app
    ```
    (Press `Ctrl+C` to stop viewing logs without stopping the app).

*   **Stop the Application**: To stop the container, run:
    ```powershell
    docker stop supply-chain-app
    ```

