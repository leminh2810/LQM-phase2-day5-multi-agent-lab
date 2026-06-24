"""Writer agent skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`.

        Synthesizes research and analysis notes into a structured final report with sources cited.
        """
        llm = LLMClient()

        sources_text = "\n\n".join(
            f"Source [{i+1}]: {src.title}\nURL: {src.url}\nContent: {src.snippet}"
            for i, src in enumerate(state.sources)
        )

        system_prompt = (
            "You are the Writer Agent in a Multi-Agent Research System.\n"
            "Your job is to synthesize a final, cohesive, and comprehensive report to answer the user's research query.\n"
            "Guidelines:\n"
            "- Integrate facts from the research notes and insights from the analysis notes.\n"
            "- Tailor the explanation for the target audience: {audience}.\n"
            "- Use inline citations (e.g. [1], [2] or [Source 1], [Source 2]) to cite sources for every factual assertion.\n"
            "- Do NOT make any claims, assertions, or comparative statements that are not directly supported by the research notes or sources. Avoid introducing unsupported figures or generalizations.\n"
            "- Include a 'Sources' section at the end listing all the URLs and titles of the cited sources."
        ).format(audience=state.request.audience)

        feedback_text = ""
        if state.errors:
            feedback_text = "\n\nCRITICAL FEEDBACK FROM PREVIOUS DRAFT TO ADDRESS:\n" + "\n".join(
                f"- {err}" for err in state.errors
            )

        user_prompt = (
            f"User Query: {state.request.query}\n\n"
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            f"Analysis Notes:\n{state.analysis_notes or 'None'}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"{feedback_text}\n\n"
            "Synthesize the final report, ensuring you address all the critical feedback if provided."
        )

        # Clear errors now that they have been incorporated into the revision prompt
        state.errors = []

        cost = 0.0
        try:
            response = llm.complete(system_prompt, user_prompt)
            cost = response.cost_usd or 0.0
            state.final_answer = response.content.strip()
        except Exception as e:
            state.final_answer = f"Final answer synthesis failed: {e}"

        state.add_trace_event("writer_answer_synthesized", {})
        state.agent_results.append(AgentResult(
            agent=AgentName.WRITER,
            content="Drafted final report incorporating research and critical analysis with inline citations.",
            metadata={"cost_usd": cost}
        ))
        return state

