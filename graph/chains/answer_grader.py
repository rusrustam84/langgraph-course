from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

class GradeAnswer(BaseModel):
    """Binary score for generated answer correctness."""
    binary_score: str = Field(description="Answer addresses the question correctly or not, 'yes' or 'no'")

structured_llm_grader = llm.with_structured_output(GradeAnswer)

system = """You are an AI assistant that evaluates the correctness of answers to questions. Your task is to determine if the provided answer correctly addresses the question. Respond with 'yes' if the answer is correct, and 'no' if it is incorrect."""

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: {question} \n\n LLM generated answer: {generation}"),
    ]
)

answer_grader = answer_prompt | structured_llm_grader