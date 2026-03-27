import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import LegalGraphState
from rag_setup import constitution_retriever
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0.2, model_name="llama-3.1-8b-instant")

def suggest_legal_solution(state: LegalGraphState):
    summary = state["summary"]
    
    retrieved_docs = constitution_retriever.invoke(summary)
    rag_context = "\n".join([doc.page_content for doc in retrieved_docs])
    
    prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an elite Legal Strategist and Expert Counsel in Indian Law. 
Your objective is to analyze the provided case summary alongside the retrieved legal context to formulate a highly accurate, practical, and objective legal solution.

The case may involve constitutional rights, civil disputes, criminal offenses, corporate law, or administrative issues. Adapt your analysis to fit the specific facts and context provided.

You MUST structure your response strictly using the following headings:
1. 📌 Key Legal Issues: (Identify the core disputes, violations, or questions of law)
2. ⚖️ Applicable Law & Interpretation: (Explain exactly how the provided legal context applies to the case)
3. 🛠️ Suggested Recourse & Strategy: (Provide actionable, step-by-step legal remedies or defense strategies)
4. ⚠️ Risk Assessment & Caveats: (Briefly highlight any legal risks, burdens of proof, or exceptions)

Maintain a professional, authoritative, and objective legal tone. Do not invent laws; rely strictly on the provided context and established Indian legal principles."""),
    ("human", "Case Summary:\n{summary}\n\nRetrieved Legal Context:\n{context}\n\nGenerate the structured legal analysis:")
])
    
    chain = prompt | llm
    response = chain.invoke({"summary": summary, "context": rag_context})
    
    return {"rag_context": rag_context, "legal_solution": response.content, "current_step": "opportunities"}
