from dotenv import load_dotenv

from graph.chains.hallucination_grader import hallucination_grader
from graph.chains.hallucination_grader import GradeHallucinations
from graph.chains.router import question_router, RouterQuery

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

def test_hallucination_grader_answer_answer_is_grounded_yes() -> None:
    question = "agent memory"
    docs = get_retriever().invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score == "yes"


def test_hallucination_grader_answer_answer_is_hallucinating_no() -> None:
    question = "agent memory"
    docs = get_retriever().invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert res.binary_score == "no"

def test_question_router_answer_vectorstore() -> None:
    question = "agent memory"
    res: RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"

def test_question_router_answer_websearch() -> None:
    question = "how to make pizza"
    res: RouterQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"