from dotenv import load_dotenv
from typing import Any, Dict
from langchain_core.documents import Document
from langchain_community.tools.tavily_search import TavilySearchResults
from graph.state import GraphState

load_dotenv()

web_search_tool = TavilySearchResults(k=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("--WEB SEARCH")

    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})

    joined_tavily_result = "\n".join(
        [res["content"] for res in tavily_results]
    )

    web_result = Document(page_content=joined_tavily_result)

    if documents is not None:
        documents.append(web_result)
    else:
        documents = [web_result]

    return {"documents": documents, "question": question}




