import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import LegalGraphState

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

llm = ChatGroq(temperature=0.3, model_name="llama-3.1-8b-instant", api_key=groq_api_key)
search_tool = TavilySearchResults(max_results=4, tavily_api_key=tavily_api_key)

def suggest_opportunities(state: LegalGraphState):
    age = str(state.get("user_age") or "Unknown").strip()
    edu = str(state.get("user_education") or "Unknown").strip()
    spec = str(state.get("user_specialization") or "Unknown").strip()
    loc = str(state.get("user_state") or "India").strip()

    if edu == "Unknown" and spec == "Unknown":
        return {
            "recommended_policies": "",
            "recommended_exams_jobs": "",
            "current_step": "report_generation",
        }

    policy_query = f"Government of India schemes, scholarships, or policies for {edu} students in {spec} from {loc}"
    policy_results = search_tool.invoke({"query": policy_query})

    career_query = f"Upcoming government exams and tech jobs in India for {spec} graduates age {age}"
    career_results = search_tool.invoke({"query": career_query})

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful Career and Policy Advisor for Indian citizens.
        Based on the user's profile and the live search results, format two clean, distinct sections:
        1. **Beneficial Government Policies & Schemes** (Include actionable steps and website links if available).
        2. **Relevant Government Exams & Job Opportunities** (Tailored to their education and field).

        Keep the tone encouraging, accessible, and completely free of AI or technical jargon.""",
        ),
        (
            "human",
            "User Profile: Age {age}, Education: {edu} in {spec}, Location: {loc}\n\nPolicy Data: {policy_data}\n\nCareer Data: {career_data}",
        ),
    ])

    chain = prompt | llm
    response = chain.invoke(
        {
            "age": age,
            "edu": edu,
            "spec": spec,
            "loc": loc,
            "policy_data": str(policy_results),
            "career_data": str(career_results),
        }
    )

    return {
        "recommended_policies": response.content,
        "recommended_exams_jobs": response.content,
        "current_step": "report_generation",
    }
