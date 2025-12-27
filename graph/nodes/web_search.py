from dotenv import load_dotenv
from typing import Any, Dict
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearch(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB SEARCH")

    question = state["question"]
    documents = state.get("documents", [])
    search_count = state.get("search_count", 0)

    tavily_results = web_search_tool.invoke({"query": question})
    tavily_results = tavily_results["results"]

    joined_tavily_result = "\n".join(
        [res["content"] for res in tavily_results]
    )

    web_result = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]

    return {"documents": documents, "question": question, "search_count": search_count + 1}




