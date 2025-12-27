# LangGraph Course: Advanced RAG with Self-Correction

This project implements an advanced Retrieval-Augmented Generation (RAG) system using **LangGraph**. It features self-correction mechanisms, including document relevance grading, hallucination detection, and fallback to web search when retrieved documents are insufficient.

## Project Overview

The system uses a state-graph to manage the flow of a RAG application. It intelligently decides whether to generate an answer based on retrieved documents or to perform a web search to gather more information. It also validates the generated answer against the source documents and the original question.

## Process Flow

The following diagram illustrates the workflow of the system:

![Process Flow](graph.png)

### Workflow Steps:
1.  **Retrieve**: Fetches relevant documents from a Pinecone vector store based on the user's question.
2.  **Grade Documents**: Evaluates the relevance of each retrieved document. If any document is found irrelevant, the system flags the need for a web search.
3.  **Decide to Generate**: 
    *   If all documents are relevant, it proceeds to **Generate**.
    *   If any document is irrelevant, it proceeds to **Web Search**.
4.  **Web Search**: Uses the Tavily search engine to find additional information online, which is then added to the document collection.
5.  **Generate**: Uses an LLM to generate an answer based on the collected documents and the question.
6.  **Grade Generation**:
    *   **Hallucination Grader**: Checks if the generation is grounded in the provided documents. If not, it re-runs the generation.
    *   **Answer Grader**: Checks if the generation actually addresses the user's question. If it's grounded but doesn't answer the question, it performs a web search to find better information.

## Project Structure

- `graph/`: Contains the core logic of the LangGraph implementation.
    - `nodes/`: Individual steps in the graph (Retrieve, Grade, Generate, Web Search).
    - `chains/`: LangChain sequences for specific tasks (Grading, Generation).
    - `state.py`: Defines the `GraphState` shared between nodes.
    - `graph.py`: Compiles the workflow into a runnable application.
- `ingestion.py`: Script for loading and indexing documents into the Pinecone vector store.
- `main.py`: Entry point to run the application.

## Setup and Installation

1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    uv sync
    ```
3.  **Configure Environment Variables**:
    Create a `.env` file in the root directory and add your API keys:
    ```env
    OPENAI_API_KEY=your_openai_api_key
    TAVILY_API_KEY=your_tavily_api_key
    PINECONE_API_KEY=your_pinecone_api_key
    ```
    *Note: Ensure `.env` is never committed to your repository.*

## Usage

To run the RAG agent:

```bash
python main.py
```

## Security Note

This project is configured to use environment variables for sensitive information like API keys. A `.gitignore` file is included to prevent the `.env` file and other sensitive data from being pushed to version control. Always double-check that no secrets are hardcoded in the source code before pushing.
