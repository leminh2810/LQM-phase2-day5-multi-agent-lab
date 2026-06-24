# Design Template

## Problem

Xây dựng một hệ thống trợ lý nghiên cứu (Research Assistant) có khả năng nhận câu hỏi nghiên cứu dài từ người dùng, tự động tìm kiếm thông tin từ nhiều nguồn, phân tích và đánh giá chất lượng bằng chứng, rồi tổng hợp thành một báo cáo hoàn chỉnh có trích dẫn nguồn.

## Why multi-agent?

Single-agent gặp hạn chế khi phải xử lý nhiều bước phức tạp cùng lúc: tìm kiếm, phân tích, viết, và kiểm chứng. Một agent duy nhất dễ bị mất context khi prompt quá dài, không thể chuyên biệt hóa từng bước, và không có cơ chế kiểm tra chéo (critic) để đảm bảo chất lượng. Multi-agent cho phép:
- **Chuyên biệt hóa**: Mỗi agent tập trung vào một nhiệm vụ duy nhất, prompt ngắn gọn và hiệu quả hơn.
- **Kiểm tra chéo**: Critic agent đánh giá output của Writer, tăng độ chính xác.
- **Linh hoạt điều phối**: Supervisor có thể quyết định lặp lại bước nào nếu kết quả chưa đạt.

## Agent roles

| Agent | Responsibility | Input | Output | Failure mode |
|---|---|---|---|---|
| Supervisor | Điều phối workflow, quyết định agent tiếp theo, kiểm tra max iterations | ResearchState (query, notes, history) | route_history (next agent) | LLM trả sai JSON → fallback rule-based routing |
| Researcher | Tìm kiếm thông tin, sinh query, lọc nguồn, tổng hợp ghi chú | query, max_sources | sources[], research_notes | Search API fail → LLM simulated search fallback |
| Analyst | Phân tích ghi chú nghiên cứu, đánh giá bằng chứng, tìm lỗ hổng | research_notes, sources | analysis_notes | LLM fail → error message stored |
| Writer | Tổng hợp báo cáo cuối cùng với inline citations | research_notes, analysis_notes, sources | final_answer | LLM fail → error message stored |
| Critic | Kiểm tra factuality, citation coverage, format | final_answer, sources, research_notes | status (APPROVED/NEEDS_REVISION), errors[] | LLM JSON parse fail → default NEEDS_REVISION |

## Shared state

- `request: ResearchQuery` — Câu hỏi nghiên cứu gốc và các tham số (max_sources, audience)
- `iteration: int` — Đếm số bước đã thực hiện để enforce max_iterations guardrail
- `route_history: list[str]` — Lịch sử routing decisions, giúp Supervisor biết flow đã đi qua đâu
- `sources: list[SourceDocument]` — Danh sách tài liệu nguồn thu thập được, dùng chung giữa Researcher, Analyst, Writer
- `research_notes: str | None` — Ghi chú nghiên cứu tổng hợp từ Researcher
- `analysis_notes: str | None` — Kết quả phân tích từ Analyst
- `final_answer: str | None` — Báo cáo cuối cùng từ Writer
- `agent_results: list[AgentResult]` — Log kết quả từng agent, bao gồm metadata (cost_usd)
- `trace: list[dict]` — Trace events cho observability
- `errors: list[str]` — Danh sách lỗi/feedback từ Critic

## Routing policy

```
                    ┌──────────────────┐
                    │   START          │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
               ┌───│   Supervisor     │◄──────────────────┐
               │   └────────┬─────────┘                   │
               │            │ (conditional routing)       │
               │   ┌────────┼────────┬──────────┐         │
               │   ▼        ▼        ▼          ▼         │
           ┌───────┐  ┌─────────┐ ┌──────┐ ┌────────┐    │
           │Researcher│ │Analyst │ │Writer│ │Critic  │    │
           └───┬───┘  └────┬────┘ └──┬───┘ └───┬────┘    │
               │           │         │         │          │
               └───────────┴─────────┴─────────┘          │
                             │                            │
                    (all route back to Supervisor)─────────┘
                             │
                    ┌────────▼─────────┐
                    │   DONE / END     │
                    └──────────────────┘
```

Luồng chính: Supervisor → Researcher → Supervisor → Analyst → Supervisor → Writer → Supervisor → Critic → Supervisor → Done

## Guardrails

- Max iterations: 6 (configurable via `MAX_ITERATIONS` env var)
- Timeout: 60 seconds per LLM call (configurable via `TIMEOUT_SECONDS` env var)
- Retry: 3 lần với exponential backoff (2s, 4s, 8s) cho mỗi LLM call (via tenacity)
- Fallback: Supervisor sử dụng rule-based routing nếu LLM JSON parse thất bại; SearchClient sử dụng LLM-simulated search nếu Tavily API không khả dụng
- Validation: Pydantic schemas cho tất cả input/output; Critic agent kiểm tra factuality và citation coverage

## Benchmark plan

| Query | Metric | Expected Outcome |
|---|---|---|
| "Research GraphRAG state-of-the-art and write a 500-word summary" | Latency | Baseline < 30s, Multi-agent < 120s |
| | Cost (USD) | Baseline < $0.01, Multi-agent < $0.05 |
| | Quality Score (0-10) | Baseline ~5-6, Multi-agent ~7-9 |
| | Citation Coverage | Baseline ~30-50%, Multi-agent ~70-90% |
| | Success Rate | Both 100% |
