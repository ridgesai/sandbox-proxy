from __future__ import annotations


def test_cost_tracker_initial_state(proxy_modules: dict[str, object]) -> None:
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=1.0)
    assert tracker.total_cost() == 0.0
    assert tracker.within_budget() is True


def test_cost_tracker_add_usage_increments(proxy_modules: dict[str, object]) -> None:
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=1.0)
    added = tracker.add_usage(
        model="qwen/qwen3-coder-next",
        prompt_tokens=1000,
        completion_tokens=500,
        cost=0.0012,
    )
    assert added == 0.0012
    assert tracker.total_cost() == 0.0012


def test_cost_tracker_budget_exceeded(proxy_modules: dict[str, object]) -> None:
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=0.000001)
    tracker.add_usage(
        model="qwen/qwen3-coder-next",
        prompt_tokens=5000,
        completion_tokens=5000,
        cost=0.005,
    )
    assert tracker.within_budget() is False


def test_cost_tracker_budget_remaining(proxy_modules: dict[str, object]) -> None:
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=1.0)
    tracker.add_usage(
        model="qwen/qwen3-coder-next", prompt_tokens=100, completion_tokens=50, cost=0.3
    )
    assert abs(tracker.budget_remaining() - 0.7) < 1e-9


def test_cost_tracker_zero_cost_stays_within_budget(
    proxy_modules: dict[str, object],
) -> None:
    """Models that return cost=0.0 (e.g. free tier) should not exhaust the budget."""
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=1.0)
    for _ in range(10):
        tracker.add_usage(
            model="openrouter/free", prompt_tokens=1000, completion_tokens=500, cost=0.0
        )
    assert tracker.within_budget() is True


def test_cost_tracker_dump_shape(proxy_modules: dict[str, object]) -> None:
    cost_mod = proxy_modules["cost_tracker"]
    tracker = cost_mod.CostTracker(evaluation_run_id="run-1", max_cost_usd=1.0)
    tracker.add_usage(
        model="qwen/qwen3-coder-next", prompt_tokens=10, completion_tokens=5, cost=0.001
    )
    dump = tracker.dump()
    assert dump["evaluation_run_id"] == "run-1"
    assert "total_cost_usd" in dump
    assert "budget_remaining_usd" in dump
    assert "requests" in dump
    assert dump["requests"] == 1
