"""Researcher agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import json
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`.

        Generates queries, executes search, deduplicates sources, and synthesizes research notes.
        """
        llm = LLMClient()
        search_client = SearchClient()

        # Step 1: Generate search queries
        query_gen_sys = (
            "You are the query generation module of a Researcher Agent.\n"
            "Based on the user's research query, generate up to 3 distinct search queries to gather comprehensive information.\n"
            "Output the queries ONLY as a JSON list of strings. Do not include markdown formatting."
        )
        query_gen_user = f"User Query: {state.request.query}"

        queries = [state.request.query]  # Default to original query
        cost = 0.0
        try:
            response = llm.complete(query_gen_sys, query_gen_user)
            cost += response.cost_usd or 0.0
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            
            parsed_queries = json.loads(raw_content)
            if isinstance(parsed_queries, list) and all(isinstance(q, str) for q in parsed_queries):
                queries = parsed_queries
        except Exception as e:
            print(f"Query generation failed: {e}. Using user query directly.")

        # Step 2: Run searches
        new_sources = []
        seen_urls = {src.url for src in state.sources if src.url}
        for q in queries:
            results = search_client.search(q, max_results=state.request.max_sources)
            for res in results:
                if not res.url or res.url not in seen_urls:
                    new_sources.append(res)
                    if res.url:
                        seen_urls.add(res.url)

        # Update state sources
        state.sources.extend(new_sources)

        # Step 3: Synthesize research notes
        sources_text = "\n\n".join(
            f"Source [{i+1}]: {src.title}\nURL: {src.url}\nContent: {src.snippet}"
            for i, src in enumerate(state.sources)
        )

        research_sys = (
            "You are the Researcher Agent. Your task is to analyze the gathered search results "
            "and synthesize objective, detailed research notes regarding the user's query.\n"
            "Focus on extracting core technical facts, key concepts, definitions, and facts.\n"
            "Clearly reference sources when compiling facts (e.g., using [Source 1], [Source 2])."
        )
        research_user = (
            f"User Query: {state.request.query}\n\n"
            f"Search Results:\n{sources_text}\n\n"
            "Compile detailed research notes based on the search results."
        )

        try:
            response = llm.complete(research_sys, research_user)
            cost += response.cost_usd or 0.0
            state.research_notes = response.content.strip()
        except Exception as e:
            state.research_notes = f"Research notes synthesis failed: {e}"

        state.add_trace_event("researcher_notes_synthesized", {"queries_used": queries, "sources_count": len(state.sources)})
        state.agent_results.append(AgentResult(
            agent=AgentName.RESEARCHER,
            content=f"Synthesized research notes from {len(state.sources)} sources using queries: {queries}.",
            metadata={"queries": queries, "cost_usd": cost}
        ))
        return state

