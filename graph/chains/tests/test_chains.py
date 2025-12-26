from dotenv import load_dotenv

load_dotenv()

from graph.chains.retrieval_grader import  GradeDocument, retrieval_grader
from ingestion import get_retriever
from pprint import pprint
from graph.chains.generation import generation_chain

def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = get_retriever().invoke(question)
    doc_text = docs[0].page_content

    res: GradeDocument = retrieval_grader.invoke({"document": doc_text, "question": question})
    assert res.binary_score == "yes"

def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = get_retriever().invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocument = retrieval_grader.invoke({"document": doc_text, "question": "how to make pizza"})

    assert res.binary_score == "no"

def test_generation_chain() -> None:
    question = "agent memory"
    docs = get_retriever().invoke(question)
    doc_text = docs[0].page_content

    res = generation_chain.invoke({"context": doc_text, "question": question})
    pprint(res)
    assert res is not None