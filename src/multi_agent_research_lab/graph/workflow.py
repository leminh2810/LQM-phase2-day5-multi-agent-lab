"""LangGraph workflow implementation."""

from __future__ import annotations

from typing import Any

from langgraph.graph import StateGraph, END

from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.agents import (
    SupervisorAgent,
    ResearcherAgent,
    AnalystAgent,
    WriterAgent,
    CriticAgent,
)


def _to_state(data: Any) -> ResearchState:
    """Convert graph output (dict or ResearchState) back to ResearchState safely."""
    if isinstance(data, ResearchState):
        return data
    if isinstance(data, dict):
        # Filter out any LangGraph internal keys (like '__end__')
        valid_fields = set(ResearchState.model_fields.keys())
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return ResearchState(**filtered)
    raise TypeError(f"Unexpected state type: {type(data)}")


def _supervisor_node(state: ResearchState) -> ResearchState:
    return SupervisorAgent().run(state)


def _researcher_node(state: ResearchState) -> ResearchState:
    return ResearcherAgent().run(state)


def _analyst_node(state: ResearchState) -> ResearchState:
    return AnalystAgent().run(state)


def _writer_node(state: ResearchState) -> ResearchState:
    return WriterAgent().run(state)


def _critic_node(state: ResearchState) -> ResearchState:
    return CriticAgent().run(state)


def _route_decision(state: ResearchState) -> str:
    """Route from supervisor to the next agent or END."""
    if not state.route_history:
        return "supervisor"
    next_agent = state.route_history[-1]
    if next_agent == "done":
        return END
    return next_agent


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph using LangGraph."""

    def __init__(self) -> None:
        self._compiled_graph: Any = None

    def build(self) -> Any:
        """Create and compile a LangGraph graph."""
        workflow = StateGraph(ResearchState)

        # Add nodes
        workflow.add_node("supervisor", _supervisor_node)
        workflow.add_node("researcher", _researcher_node)
        workflow.add_node("analyst", _analyst_node)
        workflow.add_node("writer", _writer_node)
        workflow.add_node("critic", _critic_node)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Conditional routing from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            _route_decision,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                "critic": "critic",
                END: END,
            },
        )

        # All workers route back to supervisor
        workflow.add_edge("researcher", "supervisor")
        workflow.add_edge("analyst", "supervisor")
        workflow.add_edge("writer", "supervisor")
        workflow.add_edge("critic", "supervisor")

        self._compiled_graph = workflow.compile()
        return self._compiled_graph

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        if self._compiled_graph is None:
            self.build()

        # Convert Pydantic model to dict for LangGraph input
        input_data = state.model_dump()
        final_state_data = self._compiled_graph.invoke(input_data)
        return _to_state(final_state_data)


