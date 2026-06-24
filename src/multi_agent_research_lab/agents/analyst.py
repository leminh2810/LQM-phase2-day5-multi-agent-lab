"""Analyst agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`.

        Extracts key claims, evaluates evidence, and flags potential gaps.
        """
        llm = LLMClient()

        sources_text = "\n\n".join(
            f"Source [{i+1}]: {src.title}\nURL: {src.url}\nContent: {src.snippet}"
            for i, src in enumerate(state.sources)
        )

        system_prompt = (
            "You are the Analyst Agent in a Multi-Agent Research System.\n"
            "Your job is to critically analyze the compiled research notes and underlying source documents.\n"
            "Analyze the information under the following criteria:\n"
            "1. Core Claims: What are the key assertions or technical facts discovered?\n"
            "2. Evidence Evaluation: Are the claims supported by the search sources? Note any weak assertions.\n"
            "3. Contradictions & Gaps: Identify any conflicting details, missing viewpoints, or areas needing further clarification.\n\n"
            "Format your response as structured, detailed analysis notes."
        )

        user_prompt = (
            f"User Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or 'No research notes available.'}\n\n"
            f"Sources Evaluated:\n{sources_text}\n\n"
            "Generate the structured analysis notes."
        )

        cost = 0.0
        try:
            response = llm.complete(system_prompt, user_prompt)
            cost = response.cost_usd or 0.0
            state.analysis_notes = response.content.strip()
        except Exception as e:
            state.analysis_notes = f"Analysis notes generation failed: {e}"

        state.add_trace_event("analyst_notes_generated", {})
        state.agent_results.append(AgentResult(
            agent=AgentName.ANALYST,
            content="Generated critical analysis notes assessing key claims and evidence.",
            metadata={"cost_usd": cost}
        ))
        return state

