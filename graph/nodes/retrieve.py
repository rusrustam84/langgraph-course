from typing import Any, Dict
from graph.state import GraphState
from ingestion import digest

def retrieve(state: GraphState) -> Dict[str, Any]:
    print("--RETRIEVE")

    question = state["question"]
    documents = digest().vectorstore.as_retriever().invoke(question)
    return {"documents": documents, "question": question}