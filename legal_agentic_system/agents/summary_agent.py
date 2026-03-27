import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from state import LegalGraphState
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

def summarize_document(state: LegalGraphState):
    raw_text = state["raw_text"]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal clerk. Extract the core legal dispute, parties involved, and key facts from the following text. Keep it objective and concise."),
        ("human", "Legal Document: \n\n{text}")
    ])
    
    chain = prompt | llm
    # Truncate text slightly to ensure it fits in the context window
    response = chain.invoke({"text": raw_text[:20000]}) 
    
    return {"summary": response.content, "current_step": "legal_analysis"}
