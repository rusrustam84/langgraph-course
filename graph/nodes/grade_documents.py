from typing import Any, Dict
from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determinize whether the retrieved documents are relevant for the question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (GraphState): The current state of the graph.

    Returns:
        Dict[str, Any]: Filtered out irrelevant documents and updated web_search state
    """

    print("--GRADE DOCUMENTS -> CHECK DOCUMENTS RELEVANCE")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke({"document": d, "question": question})
        grade = score.binary_score
        if grade.lower() == "yes":
            print(f"Document {d} is relevant for the question {question}")
            filtered_docs.append(d)
        else:
            print(f"Document {d} is not relevant for the question {question}")
            web_search = True
            continue
    return {"documents": filtered_docs, "web_search": web_search, "question": question}