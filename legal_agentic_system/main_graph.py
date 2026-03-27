from langgraph.graph import StateGraph, END
from state import CitizenState
from agents.data_agent import process_document
from agents.legal_pipeline import analyze_legal_case
from agents.profile_pipeline import analyze_profile_opportunities

# --- GRAPH 1: Legal Dashboard ---
def build_legal_graph():
    workflow = StateGraph(CitizenState)
    workflow.add_node("data_extraction", process_document)
    workflow.add_node("legal_analysis", analyze_legal_case)

    workflow.set_entry_point("data_extraction")
    workflow.add_edge("data_extraction", "legal_analysis")
    workflow.add_edge("legal_analysis", END)

    return workflow.compile()


# --- GRAPH 2: Profile Dashboard ---
def build_profile_graph():
    workflow = StateGraph(CitizenState)
    workflow.add_node("profile_analysis", analyze_profile_opportunities)

    workflow.set_entry_point("profile_analysis")
    workflow.add_edge("profile_analysis", END)

    return workflow.compile()

legal_graph = build_legal_graph()
profile_graph = build_profile_graph()
