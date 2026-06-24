"""Optional critic agent skeleton for bonus work."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import json
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking and safety-review agent."""

    name = "critic"

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and append findings to `state.errors`.

        Fact-checks the writer's output, verifies citation coverage, and reports issues.
        """
        llm = LLMClient()

        sources_text = "\n\n".join(
            f"Source [{i+1}]: {src.title}\nURL: {src.url}\nContent: {src.snippet}"
            for i, src in enumerate(state.sources)
        )

        system_prompt = (
            "You are the Critic Agent in a Multi-Agent Research System.\n"
            "Your job is to perform a rigorous review on the final report drafted by the Writer.\n"
            "Evaluate based on:\n"
            "1. Factuality: Are there unsupported assertions or contradictions with the sources?\n"
            "2. Citation Coverage: Are key claims mapped to inline source references?\n"
            "3. Formatting & Clarity: Is the report well-structured, targeting the intended audience, and including a sources list?\n\n"
            "Guidelines:\n"
            "- Be constructive and reasonable. If the report is high-quality and the factual claims are generally supported by the sources, select 'APPROVED'.\n"
            "- If this is review number 2 or greater, you should be extremely lenient and select 'APPROVED' unless there are major factual falsehoods. Do not hold the writer to an impossible standard.\n"
            "- Only choose 'NEEDS_REVISION' if there are significant hallucinations, severe lack of citations for key facts, or critical omissions.\n\n"
            "You must return a JSON object with keys:\n"
            "- 'status': either 'APPROVED' or 'NEEDS_REVISION'\n"
            "- 'feedback': summary of review findings\n"
            "- 'errors': list of specific errors, contradictions, or missing citations found (empty list if approved).\n"
            "Do not wrap your output in markdown code block formatting (like ```json), output raw JSON."
        )

        critic_count = sum(1 for r in state.route_history if r == "critic")
        user_prompt = (
            f"User Query: {state.request.query}\n\n"
            f"Sources:\n{sources_text}\n\n"
            f"Research Notes:\n{state.research_notes or 'None'}\n\n"
            f"Final Report:\n{state.final_answer or 'None'}\n\n"
            f"This is critic review number: {critic_count + 1}.\n"
            f"Current workflow iteration: {state.iteration}.\n\n"
            "Perform fact-checking and review."
        )

        status = "NEEDS_REVISION"
        feedback = "Critic run failed to parse response."
        errors = ["Critic processing error"]

        cost = 0.0
        try:
            response = llm.complete(system_prompt, user_prompt)
            cost = response.cost_usd or 0.0
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            
            data = json.loads(raw_content)
            status = data.get("status", "NEEDS_REVISION")
            feedback = data.get("feedback", "")
            errors = data.get("errors", [])
        except Exception as e:
            print(f"Critic LLM evaluation failed: {e}")
            feedback = f"Critic parsing error: {e}"
            errors = [feedback]

        if status == "NEEDS_REVISION" and errors:
            state.errors.extend(errors)
        else:
            # If approved, we can clear errors to allow supervisor to finish
            state.errors = []

        state.add_trace_event("critic_review_completed", {"status": status, "errors_count": len(errors)})
        state.agent_results.append(AgentResult(
            agent=AgentName.CRITIC,
            content=f"Status: {status}. Feedback: {feedback}",
            metadata={"status": status, "errors": errors, "cost_usd": cost}
        ))
        return state

