"""Benchmark skeleton for single-agent vs multi-agent."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState


Runner = Callable[[str], ResearchState]


import json
from multi_agent_research_lab.services.llm_client import LLMClient


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, calculate total cost, and run an LLM judge to evaluate quality and citations."""
    started = perf_counter()
    
    # Run the agent (baseline or multi-agent)
    try:
        state = runner(query)
        success = 1.0 if (state.final_answer and not state.errors) else 0.0
    except Exception as e:
        print(f"Runner failed with error: {e}")
        # Create a dummy state with error
        from multi_agent_research_lab.core.schemas import ResearchQuery
        state = ResearchState(request=ResearchQuery(query=query))
        state.errors.append(str(e))
        success = 0.0

    latency = perf_counter() - started

    # Calculate total cost from metadata
    total_cost = 0.0
    for res in state.agent_results:
        total_cost += res.metadata.get("cost_usd", 0.0)

    quality_score = 0.0
    citation_coverage = 0.0
    notes = "Evaluation failed or no answer generated."

    # Use LLM Judge to evaluate quality and citations if we have a final answer
    if state.final_answer:
        llm = LLMClient()
        eval_sys = (
            "You are an AI research output evaluator.\n"
            "Your job is to grade the final answer of a research assistant.\n"
            "Evaluate two metrics:\n"
            "1. quality_score: A rating between 0.0 and 10.0 based on depth of content, readability, accuracy, and structure.\n"
            "2. citation_coverage: A float between 0.0 and 1.0 representing the proportion of factual statements that are properly cited using inline citations (like [1], [Source 1], etc.).\n\n"
            "You must output your evaluation ONLY as a JSON object with the following keys:\n"
            "- 'quality_score': float (0.0 to 10.0)\n"
            "- 'citation_coverage': float (0.0 to 1.0)\n"
            "- 'notes': a short summary of your evaluation comments.\n"
            "Do not wrap in markdown, output raw JSON."
        )
        eval_user = (
            f"User Query: {query}\n\n"
            f"Final Report:\n{state.final_answer}\n\n"
            "Please evaluate."
        )

        try:
            response = llm.complete(eval_sys, eval_user)
            raw_content = response.content.strip()
            if raw_content.startswith("```"):
                lines = raw_content.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                raw_content = "\n".join(lines).strip()
            
            eval_data = json.loads(raw_content)
            quality_score = float(eval_data.get("quality_score", 0.0))
            citation_coverage = float(eval_data.get("citation_coverage", 0.0))
            notes = eval_data.get("notes", "")
            # Accumulate evaluation LLM cost as well to keep the cost benchmarking exact
            total_cost += response.cost_usd or 0.0
        except Exception as e:
            print(f"LLM Judge evaluation failed: {e}")
            notes = f"LLM Judge evaluation failed: {e}"

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=total_cost,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        success_rate=success,
        notes=notes
    )

    return state, metrics

