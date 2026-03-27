import os
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults

load_dotenv()
tavily_api_key = os.getenv("TAVILY_API_KEY")

gov_search_tool = TavilySearchResults(
    max_results=4,
    search_depth="advanced",
    include_domains=["gov.in", "nic.in", "india.gov.in"],
    tavily_api_key=tavily_api_key,
)
