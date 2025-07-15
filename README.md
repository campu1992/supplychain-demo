# Smart Supply Chain Analysis Agent

## Quick Start: Running with Docker

This is the recommended and most straightforward way to run the project.

### Prerequisites

*   **Docker:** You must have Docker installed and running.
*   **Gemini API Key:** You need a valid Google Gemini API key.

### Instructions

#### Step 1: Configure your API Key
1.  Open the `config_new.yaml` file in the project root.
2.  Replace the placeholder `YOUR_GEMINI_API_KEY` with your actual API key. The application will not start without it.

#### Step 2: Build the Docker Image (One-Time Setup)
Open your terminal (PowerShell, Bash, etc.) in the project's root directory and run this command. It packages the entire application into a self-contained image.

```bash
docker build -t supply-chain-agent .
```

#### Step 3: Build the Vector Database (One-Time Setup)
Next, run this command to process the source data and create the vector database. It runs the indexing script inside a container and saves the database into a `chroma_data` folder on your local machine, so it persists between runs.

```bash
# In PowerShell (Windows):
docker run --rm -v "${PWD}/chroma_data:/app/chroma_data" supply-chain-agent python scripts/build_index.py

# In Bash (Linux/macOS):
docker run --rm -v "$(pwd)/chroma_data:/app/chroma_data" supply-chain-agent python scripts/build_index.py
```
> **Note:** This process will take several minutes as it needs to download models and process all the data. You only need to do this once.

#### Step 4: Launch the Application
This is the command you will use every time you want to start the application. It launches the web server and the AI agent, connecting to the database you created in the previous step.

```bash
docker run -it --rm -p 8501:8501 -v "${PWD}/chroma_data:/app/chroma_data" supply-chain-agent
```

Once running, open your web browser and navigate to **`http://localhost:8501`**.

To stop the application, press `Ctrl+C` in your terminal.

---

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
    *   **Mechanism**: This is the classic RAG pipeline we built. It takes the user's query, embeds it, retrieves relevant document chunks from the ChromaDB vector store, uses a Cross-Encoder to rerank the results for precision, and finally passes the best context to a Gemini LLM to synthesize a final answer.
    *   **Best for**: "Why did late deliveries spike?", "What were the common issues for returns?"

2.  **`quantitative_data_analysis_tool`**:
    *   **Purpose**: Handles quantitative and analytical questions that require exact calculations.
    *   **Mechanism**: This tool is a powerful `Pandas DataFrame Agent`. The agent is given access to the raw `DataCoSupplyChainDataset.csv` loaded into a Pandas DataFrame. When invoked, the LLM writes and executes Python code (using Pandas) on the fly to compute the answer directly from the source data.
    *   **Best for**: "How many orders were there for the Apparel category in May 2018?", "What was the average product price in the Sports category?"

This dual-tool approach gives the system the flexibility to provide both nuanced, context-aware narratives and precise, calculated numerical answers.

### Models Used

*   **Embedding Model (`BAAI/bge-large-en-v1.5`):** This state-of-the-art model is used to convert all documents (order data and user summaries) and incoming user queries into high-dimensional vectors (embeddings). Its high performance is crucial for the quality of the initial document retrieval.
*   **Reranking Model (`cross-encoder/ms-marco-MiniLM-L-6-v2`):** This Cross-Encoder model refines the search results from the vector store. It takes the top N retrieved documents and the user's query, directly compares them, and re-sorts them based on true relevance, significantly improving the context quality for the final answer.
*   **Core LLM (`gemini-1.5-flash`):** This powerful and efficient model from Google serves two key roles:
    1.  It is the "brain" of the agent, deciding which tool to use.
    2.  It acts as the synthesizer for the RAG tool, generating a human-readable answer from the provided context.

## 3. Challenges and Solutions

Throughout the project, several challenges were encountered and overcome:

*   **Initial `ModuleNotFoundError`s:** The environment setup was complex, leading to various import errors. **Solution:** We created a comprehensive `requirements.txt` file and used it to build a stable, reproducible environment.
*   **Semantic vs. Quantitative Questions:** The initial RAG-only approach failed on numerical questions ("how many..."). **Solution:** We evolved the architecture from a simple pipeline to a LangChain-powered agent with distinct tools for semantic search and quantitative pandas analysis.
*   **ChromaDB `NoneType` Error:** The indexing script failed because some metadata fields in the raw data were null. **Solution:** We added a data cleaning step just before batch insertion into ChromaDB to convert any non-standard data types (like `None`) into strings.
*   **Embedding Model Mismatch:** After upgrading the embedding model, a dimension mismatch error occurred (768 vs. 1024). **Solution:** We updated the project configuration (`config_new.yaml`) and refactored the `EmbeddingModel` class to ensure all components consistently used the new, higher-dimension model (`BAAI/bge-large-en-v1.5`).
*   **LangChain Agent Parsing Errors:** The agent failed because it expected its internal thought process keywords to be in English. **Solution:** We updated the agent's master prompt to use the required English keywords (`Thought`, `Action`, etc.) while keeping the explanatory text in Spanish for clarity during development.

## 4. How to Run (Local Execution)

For developers who want to run the project without Docker.

### Prerequisites

*   Python 3.11+
*   A Google Gemini API Key

### Local Execution

1.  **Clone the Repository:**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Set Up a Virtual Environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Key:**
    *   Open the `config_new.yaml` file.
    *   Replace the placeholder `"YOUR_GEMINI_API_KEY"` with your actual Google Gemini API key.

5.  **Run the ChromaDB Server:**
    *   In a separate terminal, start the ChromaDB server:
    ```bash
    chroma run --host localhost --port 8000
    ```

6.  **Build the Vector Index:**
    *   Before running the app for the first time, you must build the vector index. Run the indexing script:
    ```bash
    python scripts/build_index.py
    ```
    *   This script will process all the data, generate embeddings, and populate the ChromaDB database. It may take some time.

7.  **Launch the Application:**
    *   Once indexing is complete, run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
