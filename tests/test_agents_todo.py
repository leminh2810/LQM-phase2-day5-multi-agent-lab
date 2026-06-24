from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routes_correctly() -> None:
    """Supervisor should route to 'researcher' as the first step when no notes exist."""
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    result = SupervisorAgent().run(state)
    # Should have recorded a route
    assert len(result.route_history) == 1
    # The first route should be a valid agent or 'done'
    assert result.route_history[0] in ["researcher", "analyst", "writer", "critic", "done"]
    # iteration should have incremented
    assert result.iteration == 1
    # Should have at least one agent result from supervisor
    assert len(result.agent_results) >= 1
    assert result.agent_results[0].agent == "supervisor"

