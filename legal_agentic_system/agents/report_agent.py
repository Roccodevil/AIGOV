import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import LegalGraphState

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0.1, model_name="llama-3.1-8b-instant", api_key=groq_api_key)

def compile_final_report(state: LegalGraphState):
    summary = state.get("summary", "")
    solution = state.get("legal_solution", "")
    policies = state.get("recommended_policies", "")
    exams_jobs = state.get("recommended_exams_jobs", "")

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are an expert public servant and citizen advocate. Your job is to compile the provided legal analysis and policy recommendations into one comprehensive, easy-to-read Final Report.

        Rules:
        1. STRIP ALL JARGON: Do not use terms like "LLM", "RAG", "Agent", "Context", or "Prompt".
        2. Use clear headings, bullet points, and bold text for readability.
        3. Structure the report as follows:
           - 📝 Case Overview (Simplify the legal summary)
           - ⚖️ Actionable Legal Guidance (Simplify the legal solution)
           - 🎓 Citizen Opportunities (Display the policies, exams, and jobs)
           - 🌐 Next Steps & Support Links
        4. Speak directly to the user in a helpful, empathetic, and professional tone.""",
        ),
        (
            "human",
            "Legal Summary: {summary}\n\nLegal Solution: {solution}\n\nPolicies: {policies}\n\nExams and Jobs: {exams_jobs}",
        ),
    ])

    chain = prompt | llm
    response = chain.invoke(
        {
            "summary": summary,
            "solution": solution,
            "policies": policies,
            "exams_jobs": exams_jobs,
        }
    )

    return {"final_report": response.content, "current_step": "end"}
