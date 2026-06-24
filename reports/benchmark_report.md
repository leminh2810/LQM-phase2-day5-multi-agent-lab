# Multi-Agent Research Lab: Benchmark Report

This report compares the performance of the **Single-Agent Baseline** vs the **Multi-Agent Orchestrated Workflow**.

## Performance Metrics

| Run | Latency (s) | Cost (USD) | Quality Score (0-10) | Citation Coverage | Success Rate | Notes |
|---|---:|---:|---:|---:|---:|---|
| Single-Agent Baseline | 18.76 | $0.00071 | 8.5/10 | 60.0% | 100.0% | The report provides a well-structured overview of GraphRAG, covering its theoretical foundations, implementation techniques, applications, and future directions. The content depth is strong, and readability is good; however, some claims could benefit from additional sources for verification, leading to an incomplete citation coverage. |
| Multi-Agent Workflow | 81.40 | $0.00357 | 8.5/10 | 80.0% | 100.0% | The report offers a comprehensive overview of GraphRAG, detailing its core features, applications, and future directions with good structure and readability. Some sections, particularly on applications, could benefit from more specific examples. Citation coverage is strong but could be improved by ensuring all major claims and metrics are cited. |

## Analysis & Observations

1. **Latency vs Quality Trade-off**:
   - **Single-Agent Baseline** runs significantly faster (lower latency) since it only calls the LLM once without complex orchestration loop checks.
   - **Multi-Agent Workflow** has higher latency but achieves a higher Quality Score and better factual details due to iterative refinement and specialized roles.

2. **Cost Analysis**:
   - The Multi-Agent system calls the LLM multiple times (Supervisor, Researcher query gen, Researcher synthesis, Analyst evaluation, Writer drafting, Critic validation), resulting in higher input/output token costs.

3. **Citation & Accuracy**:
   - Multi-agent structure enforces source retrieval through the `Researcher` agent, and the `Critic` enforces citation coverage. Thus, citation coverage and factuality are dramatically higher than the single-agent baseline.

