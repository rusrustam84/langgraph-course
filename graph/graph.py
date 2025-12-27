from dotenv import load_dotenv

from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader

load_dotenv()

from langgraph.graph import END, StateGraph
from graph.consts import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEBSEARCH
from graph.state import GraphState


def decide_to_generate(state: GraphState) -> str:
    print("--DECIDE TO GENERATE - ASSESS GRADED DOCUMENTS--")

    if state["web_search"]:
        print(
            "-- DECISION: NOT ALL DOCUMENTS ARE RELEVANT. RUNNING WEB SEARCH --"
        )
        return WEBSEARCH
    else:
        print("-- DECISION: ALL DOCUMENTS ARE RELEVANT. GENERATING --")
        return GENERATE
def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--GRADE GENERATION GROUNDED IN DOCUMENTS AND QUESTION--")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    print("-- CHECKING HALLUCINATIONS --")
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    print(f"Hallucination score: {score.binary_score}")
    hallucination_grade = score.binary_score
    if hallucination_grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"generation": generation, "question": question})
        answer_grade = score.binary_score
        if answer_grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS---")
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEBSEARCH, web_search)

workflow.set_entry_point(RETRIEVE)
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEBSEARCH: WEBSEARCH,
     GENERATE: GENERATE},
)

workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEBSEARCH, "not supported": GENERATE}
)

workflow.add_edge(WEBSEARCH, GENERATE)

app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")






