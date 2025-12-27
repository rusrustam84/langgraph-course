from typing import List, TypedDict, Any

class GraphState(TypedDict):
    """State representation for the graph

    Attributes:
        question: str - The current question being asked by the user.
        generation: str - The current search query entered by the user.
        web_search: bool - Whether to run web search.
        documents: List[Any] - The list of documents retrieved from the graph.
        """
    question: str
    generation: str
    web_search: bool
    documents: List[Any]
    search_count: int