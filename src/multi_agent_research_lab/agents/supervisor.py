"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


import json
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route.

        Decides which agent to route to based on state.
        """
        settings = get_settings()

        # Iteration Guardrail
        if state.iteration >= settings.max_iterations:
            if not state.final_answer and state.research_notes and state.analysis_notes:
                next_agent = "writer"
            else:
                next_agent = "done"
            reason = f"Max iterations reached ({state.iteration} >= {settings.max_iterations}). Forcing route to: {next_agent}"
            state.record_route(next_agent)
            state.add_trace_event("supervisor_decision", {"next_agent": next_agent, "reason": reason})
            state.agent_results.append(AgentResult(
                agent=AgentName.SUPERVISOR,
                content=reason,
                metadata={"next_agent": next_agent}
            ))
            return state

        # Prompt LLM for routing decision
        system_prompt = (
            "You are the Supervisor of a Multi-Agent Research System.\n"
            "Your job is to orchestrate a workflow to solve the user's research query by routing to the appropriate agent.\n\n"
            "Available agents:\n"
            "1. 'researcher': Search websites/documents and write initial research notes. Use this agent when you need to gather information, or if the current notes are missing information to address the query.\n"
            "2. 'analyst': Extract key claims, compare viewpoints, evaluate evidence strength, and identify weak spots/gaps. Use this agent after 'researcher' has gathered notes, or if the research notes need deeper analysis.\n"
            "3. 'writer': Synthesize a comprehensive final report with citations based on research and analysis notes. Use this agent when you have both research notes and analysis notes ready to draft the final response.\n"
            "4. 'critic': Review the final answer drafted by the writer, checking for accuracy, citation coverage, or hallucinations. Use this agent after 'writer' has generated a final answer.\n"
            "5. 'done': End the workflow. Use this agent ONLY after the final answer is reviewed and verified by the 'critic', or if the current final answer is correct and requires no further edits.\n\n"
            "Guidelines:\n"
            "- Each step should be justified.\n"
            "- Follow a logical sequence: researcher -> analyst -> writer -> critic -> done.\n"
            "- If the 'writer' is the last agent in the route history, you must route to 'critic' to review the new draft.\n"
            "- If the critic has just run and reported errors (found in errors/warnings), route to the 'writer' to address them.\n"
            "- Do not loop infinitely.\n\n"
            "You must return a JSON object with two keys:\n"
            "- 'next_agent': one of ['researcher', 'analyst', 'writer', 'critic', 'done']\n"
            "- 'reason': a brief explanation of why this agent was selected.\n"
            "Do not wrap your output in markdown code block formatting (like ```json), output raw JSON."
        )

        user_prompt = (
            f"User Query: {state.request.query}\n"
            f"Current Iteration: {state.iteration}\n"
            f"Route History: {state.route_history}\n"
            f"Research Notes Present: {state.research_notes is not None}\n"
            f"Analysis Notes Present: {state.analysis_notes is not None}\n"
            f"Final Answer Present: {state.final_answer is not None}\n"
            f"Errors / Feedback: {state.errors}\n\n"
            "Please determine the next agent."
        )

        llm = LLMClient()
        next_agent = "done"
        reason = "Fallback choice due to LLM parsing error."
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
            next_agent = data.get("next_agent", "done")
            reason = data.get("reason", "")
        except Exception as e:
            # Deterministic Fallback if LLM complete or json load fails
            print(f"Supervisor LLM decision failed: {e}. Falling back to rule-based routing.")
            if not state.research_notes:
                next_agent = "researcher"
                reason = "Rule-based: no research notes gathered yet."
            elif not state.analysis_notes:
                next_agent = "analyst"
                reason = "Rule-based: no analysis notes generated yet."
            elif not state.final_answer:
                next_agent = "writer"
                reason = "Rule-based: no final answer generated yet."
            elif not state.route_history or state.route_history[-1] == "writer":
                next_agent = "critic"
                reason = "Rule-based: checking final answer."
            else:
                next_agent = "done"
                reason = "Rule-based: complete."

        # Double check validity of agent choice
        valid_agents = ["researcher", "analyst", "writer", "critic", "done"]
        if next_agent not in valid_agents:
            next_agent = "done"

        state.record_route(next_agent)
        state.add_trace_event("supervisor_decision", {"next_agent": next_agent, "reason": reason})
        state.agent_results.append(AgentResult(
            agent=AgentName.SUPERVISOR,
            content=f"Route decided: {next_agent}. Reason: {reason}",
            metadata={"next_agent": next_agent, "cost_usd": cost}
        ))
        return state

