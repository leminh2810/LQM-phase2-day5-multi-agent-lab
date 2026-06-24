"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import openai
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None



class LLMClient:
    """Provider-agnostic LLM client implementation using OpenAI."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._client: openai.OpenAI | None = None

    @property
    def client(self) -> openai.OpenAI:
        if self._client is None:
            self._client = openai.OpenAI(api_key=self.settings.openai_api_key)
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion."""
        response = self.client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            timeout=self.settings.timeout_seconds,
        )

        content = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else None
        output_tokens = usage.completion_tokens if usage else None

        cost_usd = None
        if input_tokens is not None and output_tokens is not None:
            model_name = self.settings.openai_model.lower()
            if "gpt-4o-mini" in model_name:
                cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
            elif "gpt-4o" in model_name:
                cost_usd = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
            else:
                cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )

