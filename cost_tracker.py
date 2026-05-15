from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CostRecord:
    total_cost_usd: float = 0.0
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    models: set[str] = field(default_factory=set)


class CostTracker:
    def __init__(self, evaluation_run_id: str, max_cost_usd: float):
        self.evaluation_run_id = evaluation_run_id
        self.max_cost_usd = max_cost_usd
        self._record = CostRecord()

    def add_usage(
        self,
        *,
        model: str,
        prompt_tokens: int,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ) -> float:
        """Record usage for a single request.

        `cost` is the authoritative USD amount returned by OpenRouter in
        `usage.cost`.  Token counts are kept for observability only.
        """
        self._record.total_cost_usd += cost
        self._record.requests += 1
        self._record.prompt_tokens += prompt_tokens
        self._record.completion_tokens += completion_tokens
        self._record.models.add(model)
        return cost

    def within_budget(self) -> bool:
        return self._record.total_cost_usd <= self.max_cost_usd

    def total_cost(self) -> float:
        return self._record.total_cost_usd

    def budget_remaining(self) -> float:
        return self.max_cost_usd - self._record.total_cost_usd

    def dump(self) -> dict[str, object]:
        return {
            "evaluation_run_id": self.evaluation_run_id,
            "total_cost_usd": round(self._record.total_cost_usd, 10),
            "max_cost_usd": self.max_cost_usd,
            "budget_remaining_usd": round(self.budget_remaining(), 10),
            "requests": self._record.requests,
            "prompt_tokens": self._record.prompt_tokens,
            "completion_tokens": self._record.completion_tokens,
            "models": sorted(self._record.models),
        }
