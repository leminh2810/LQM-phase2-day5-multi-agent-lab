from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery, AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.services.storage import LocalArtifactStore

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def run_baseline_workflow(query: str) -> ResearchState:
    """Run a single-agent RAG workflow."""
    request = ResearchQuery(query=query)
    state = ResearchState(request=request)
    
    llm = LLMClient()
    search_client = SearchClient()
    
    # 1. Web Search
    sources = search_client.search(query, max_results=request.max_sources)
    state.sources = sources
    
    # 2. Format Sources
    sources_text = "\n\n".join(
        f"Source [{i+1}]: {src.title}\nURL: {src.url}\nContent: {src.snippet}"
        for i, src in enumerate(sources)
    )
    
    # 3. LLM completion
    system_prompt = (
        "You are a research assistant. Provide a comprehensive report on the user query.\n"
        "Use the following search results to support your report.\n"
        "Include inline citations (like [1], [2]) and list your sources at the end."
    )
    user_prompt = f"User Query: {query}\n\nSearch Results:\n{sources_text}"
    
    response = llm.complete(system_prompt, user_prompt)
    state.final_answer = response.content.strip()
    
    state.agent_results.append(AgentResult(
        agent=AgentName.WRITER,
        content=state.final_answer,
        metadata={"cost_usd": response.cost_usd or 0.0}
    ))
    return state


def run_multi_agent_workflow(query: str) -> ResearchState:
    """Run the multi-agent workflow."""
    state = ResearchState(request=ResearchQuery(query=query))
    workflow = MultiAgentWorkflow()
    return workflow.run(state)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline with web search (RAG)."""
    _init()
    console.print(f"[bold blue]Running Single-Agent Baseline for query:[/bold blue] {query}")
    state = run_baseline_workflow(query)
    console.print(Panel.fit(state.final_answer or "No answer generated.", title="Single-Agent Baseline Answer"))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""
    _init()
    console.print(f"[bold green]Running Multi-Agent Workflow for query:[/bold green] {query}")
    try:
        state = run_multi_agent_workflow(query)
        console.print(Panel.fit(state.final_answer or "No answer generated.", title="Multi-Agent Final Answer"))
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc


@app.command()
def benchmark(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Compare single-agent vs multi-agent and output a benchmark report."""
    _init()
    console.print(f"[bold magenta]Starting Benchmark for query:[/bold magenta] {query}\n")

    # Run baseline benchmark
    console.print("[yellow]Running Single-Agent Baseline...[/yellow]")
    baseline_state, baseline_metrics = run_benchmark(
        "Single-Agent Baseline", query, run_baseline_workflow
    )

    # Run multi-agent benchmark
    console.print("[yellow]Running Multi-Agent Workflow...[/yellow]")
    multi_state, multi_metrics = run_benchmark(
        "Multi-Agent Workflow", query, run_multi_agent_workflow
    )

    # Render report
    metrics_list = [baseline_metrics, multi_metrics]
    report_md = render_markdown_report(metrics_list)
    
    # Save report using LocalArtifactStore
    store = LocalArtifactStore()
    report_path = store.write_text("benchmark_report.md", report_md)
    
    console.print(f"\n[bold green]Benchmark completed successfully![/bold green] Report saved to: {report_path}\n")
    console.print(Panel.fit(report_md, title="Benchmark Report"))


if __name__ == "__main__":
    app()

