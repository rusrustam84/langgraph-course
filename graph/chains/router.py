from dotenv import load_dotenv

from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()


class RouterQuery(BaseModel):
    """
    Represents a query for routing user questions to either a vector store or web search.
    """

    datasource: Literal["vectorstore", "websearch"] = Field(
        ..., description="Given a user question to choose to route into the vector store or web search"
    )

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

structured_llm_router = llm.with_structured_output(RouterQuery)

system = """You are a router that decides whether to route a user question to either a vector store or web search.
The vector store contains documents related to agents, prompt engineering, and adversarial attacks.
Use the vector store for questions on these topics. For all other questions, use web search.
"""

router_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)

question_router = router_prompt | structured_llm_router


