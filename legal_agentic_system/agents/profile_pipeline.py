import os
import json
import re
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import CitizenState
from agents.search_tool import gov_search_tool
from agents.policy_rag import policy_retriever

llm = ChatGroq(
    temperature=0.1,
    model_name="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY"),
)


def _extract_json_block(text: str):
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return []

    return []


def analyze_profile_opportunities(state: CitizenState):
    age = str(state.get("user_age") or "").strip()
    gender = str(state.get("user_gender") or "").strip()
    income = str(state.get("annual_income") or "").strip()
    user_status = str(state.get("current_status") or "").strip()
    edu = str(state.get("user_education") or "").strip()
    spec = str(state.get("user_specialization") or "").strip()
    loc = str(state.get("user_state") or "").strip()
    extra = str(state.get("extracurriculars") or "").strip()
    goal = str(state.get("user_goal") or "Find best suited opportunities").strip()

    profile_context = (
        f"Age: {age}, Gender: {gender}, Income: {income}, Status: {user_status}, "
        f"Education: {edu}, Specialization: {spec}, Location: {loc}, Experience: {extra}, Goal: {goal}"
    )

    rag_query = (
        f"Eligibility and rules for {user_status or 'student'} in {loc or 'India'} "
        f"regarding {goal}. Consider gender {gender} and income {income}."
    )
    rag_context = ""
    if policy_retriever is not None:
        rag_docs = policy_retriever.invoke(rag_query)
        rag_context = "\n".join([doc.page_content for doc in rag_docs])

    active_query = (
        f"Government of India active schemes, grants, and jobs for {gender} {user_status}, "
        f"{edu} in {spec}, income {income}, located in {loc}"
    )
    active_data = gov_search_tool.invoke({"query": active_query})

    innovation_query = (
        f"Government of India startup grants, innovation funds, or fellowships "
        f"for students with experience in: {extra}"
    )
    innovation_data = gov_search_tool.invoke({"query": innovation_query})

    missed_query = (
        f"Recently closed government deadlines, expired schemes, or age limits passed "
        f"for age {age} {spec} graduates in India"
    )
    missed_data = gov_search_tool.invoke({"query": missed_query})

    opportunities_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an elite AI Citizen Advisor.
Return ONLY valid JSON array with exactly 3 objects. Do not include markdown or prose.
Each object must contain:
- name
- category
- why_this_suits_you
- eligibility_fit
- average_time_to_redeem
- accessibility_score
- action_steps (array of short strings)
- official_link

Rules:
1) Use only provided RAG context and live government data.
2) Filter out schemes/jobs the user is unlikely to qualify for (age, gender, income, status).
3) Keep output strict JSON.""",
        ),
        (
            "human",
            "Profile: {profile}\n\nPolicy RAG Context:\n{rag_context}\n\nActive Data: {active}\n\nInnovation Data: {innovation}",
        ),
    ])

    opportunities_raw = (opportunities_prompt | llm).invoke(
        {
            "profile": profile_context,
            "rag_context": rag_context,
            "active": str(active_data),
            "innovation": str(innovation_data),
        }
    ).content

    opportunities_json = _extract_json_block(opportunities_raw)

    missed_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Return ONLY valid JSON array of up to 3 missed opportunities.
Each object must include:
- name
- reason_missed
- next_cycle_hint
- expected_next_window
- official_link""",
        ),
        (
            "human",
            "Profile: {profile}\n\nMissed Data: {missed}",
        ),
    ])

    missed_raw = (missed_prompt | llm).invoke(
        {
            "profile": profile_context,
            "missed": str(missed_data),
        }
    ).content
    missed_json = _extract_json_block(missed_raw)

    roadmap_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Create a concise, actionable roadmap in markdown.
Sections:
### 🎯 Core Eligibility Schemes
### 🚀 Career & Innovation Grants
### ⚠️ Missed Opportunities
Use empathetic plain language and include practical next actions.""",
        ),
        (
            "human",
            "Profile: {profile}\n\nSelected Opportunities: {opps}\n\nMissed: {missed}",
        ),
    ])

    roadmap = (roadmap_prompt | llm).invoke(
        {
            "profile": profile_context,
            "opps": json.dumps(opportunities_json, ensure_ascii=False),
            "missed": json.dumps(missed_json, ensure_ascii=False),
        }
    ).content

    return {
        "active_opportunities": str(active_data),
        "missed_opportunities": str(missed_data),
        "opportunities_json": opportunities_json,
        "missed_opportunities_json": missed_json,
        "guiding_roadmap": roadmap,
        "insight_report": roadmap,
    }
