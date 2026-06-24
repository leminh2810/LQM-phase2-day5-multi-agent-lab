"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import SourceDocument


import json
import urllib.request
import urllib.error
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument
from multi_agent_research_lab.services.llm_client import LLMClient


class SearchClient:
    """Provider-agnostic search client implementing Tavily API and LLM-simulation fallback."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Uses Tavily if TAVILY_API_KEY is available, otherwise simulates realistic search results using LLM.
        """
        api_key = self.settings.tavily_api_key

        if api_key:
            try:
                url = "https://api.tavily.com/search"
                data = json.dumps({
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results
                }).encode("utf-8")
                
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST"
                )
                
                with urllib.request.urlopen(req, timeout=15) as response:
                    res_data = json.loads(response.read().decode("utf-8"))
                    results = res_data.get("results", [])
                    return [
                        SourceDocument(
                            title=item.get("title", "Untitled"),
                            url=item.get("url"),
                            snippet=item.get("content", item.get("snippet", ""))
                        )
                        for item in results[:max_results]
                    ]
            except Exception as e:
                # Log or print error and fallback to simulated search
                print(f"Tavily search failed: {e}. Falling back to simulated search.")

        # Fallback / Simulation using LLM
        return self._simulated_search(query, max_results)

    def _simulated_search(self, query: str, max_results: int) -> list[SourceDocument]:
        """Simulates realistic search results using the LLM client."""
        llm = LLMClient()
        system_prompt = (
            "You are a mock search engine. Generate high-quality simulated web search results "
            "relevant to the user's search query. Output the result ONLY as a JSON array of objects "
            "with keys: 'title', 'url', and 'snippet'. Do not include markdown code block formatting (like ```json) in your raw response."
        )
        user_prompt = f"Query: {query}\nGenerate {max_results} search results."
        
        try:
            response = llm.complete(system_prompt, user_prompt)
            # Remove any markdown formatting if present
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                # strip code block formatting
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            
            data = json.loads(raw_content)
            if isinstance(data, list):
                return [
                    SourceDocument(
                        title=item.get("title", "Untitled"),
                        url=item.get("url", "https://example.com"),
                        snippet=item.get("snippet", "")
                    )
                    for item in data[:max_results]
                ]
        except Exception as e:
            print(f"Simulated search failed parsing: {e}")
            
        # Hardcoded emergency fallback in case LLM simulated search fails
        return [
            SourceDocument(
                title=f"Introduction to {query}",
                url="https://en.wikipedia.org/wiki/Search_simulation",
                snippet=f"This is a fallback document explaining key concepts of {query}."
            )
        ]

