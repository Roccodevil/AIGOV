import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import CitizenState
from agents.search_tool import gov_search_tool

llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
)


def analyze_legal_case(state: CitizenState):
    category = state.get("legal_category", "General")
    status = state.get("case_status", "Unknown Stage")
    raw_text = state.get("raw_text", "")

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Summarize the core legal issue or dispute from the text in 3 concise bullet points."),
        ("human", "{text}"),
    ])
    summary = (summary_prompt | llm).invoke({"text": (raw_text or "")[:10000]}).content

    search_query = (
        f"Government of India official procedure for {category} dispute "
        f"at stage: {status}. Case details: {summary}"
    )
    live_data = gov_search_tool.invoke({"query": search_query})

    solution_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Citizen Legal Advisor. Based on the user's issue and the live government data, provide a structured legal action plan.
        Do not hallucinate laws. Only use the provided live data. Include website links if present in the data.
        Format cleanly with markdown headings:
        - ⚖️ Case Assessment
        - 🛠️ Recommended Official Steps
        - 🏛️ Official Portals / Resources""",
        ),
        (
            "human",
            "Category: {category}\nStage: {status}\nUser Issue: {summary}\n\nLive Gov Data: {live_data}",
        ),
    ])

    solution = (solution_prompt | llm).invoke(
        {
            "category": category,
            "status": status,
            "summary": summary,
            "live_data": str(live_data),
        }
    ).content

    return {
        "legal_summary": summary,
        "legal_solution": solution,
        "gov_resources": str(live_data),
    }
