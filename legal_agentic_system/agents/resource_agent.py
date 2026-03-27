import os

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from state import LegalGraphState
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
search_tool = TavilySearchResults(max_results=3)

def find_gov_resources(state: LegalGraphState):
    solution = state["legal_solution"]
    
    search_query = f"official government of India portal or scheme for: {solution[:100]}"
    search_results = search_tool.invoke({"query": search_query})
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an administrative assistant. Format the following web search results into a clean list of actionable government steps, resources, and official website links. Ensure links are included if available."),
        ("human", "Search Results: {results}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"results": str(search_results)})
    
    return {"government_resources": [response.content], "current_step": "end"}
