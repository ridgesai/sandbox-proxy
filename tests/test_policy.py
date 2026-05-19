from __future__ import annotations

import pytest


def test_validate_model_allows_unrestricted_model(
    proxy_modules: dict[str, object],
) -> None:
    policy = proxy_modules["policy"]
    body = {"model": "gpt-4o"}
    policy.validate_model(body, frozenset())


def test_validate_model_rejects_restricted_model(
    proxy_modules: dict[str, object],
) -> None:
    policy = proxy_modules["policy"]
    body = {"model": "gpt-4o"}
    with pytest.raises(policy.PolicyError, match="Restricted model"):
        policy.validate_model(body, {"gpt-4o"})


def test_validate_models_array_rejects_restricted_entry(
    proxy_modules: dict[str, object],
) -> None:
    policy = proxy_modules["policy"]
    body = {
        "model": "deepseek/deepseek-r1-0528",
        "models": ["deepseek/deepseek-r1-0528", "gpt-4o"],
    }
    with pytest.raises(policy.PolicyError, match="gpt-4o"):
        policy.validate_model(body, {"gpt-4o"})


def test_validate_model_requires_selection(proxy_modules: dict[str, object]) -> None:
    policy = proxy_modules["policy"]
    with pytest.raises(policy.PolicyError, match="Missing required model selection"):
        policy.validate_model({}, frozenset())


def test_enforce_provider_overwrites_existing(proxy_modules: dict[str, object]) -> None:
    policy = proxy_modules["policy"]
    body = {"model": "deepseek/deepseek-r1-0528", "provider": {"order": ["evil"]}}
    rewritten = policy.enforce_provider(body, "deepseek/deepseek-r1-0528")
    assert rewritten["provider"] == policy.DEFAULT_PROVIDER


def test_enforce_provider_injects_when_missing(
    proxy_modules: dict[str, object],
) -> None:
    policy = proxy_modules["policy"]
    body = {"model": "deepseek/deepseek-r1-0528"}
    rewritten = policy.enforce_provider(body, "deepseek/deepseek-r1-0528")
    assert rewritten["provider"] == policy.DEFAULT_PROVIDER


def test_sanitize_chat_strips_unknown(proxy_modules: dict[str, object]) -> None:
    policy = proxy_modules["policy"]
    body = {
        "model": "deepseek/deepseek-r1-0528",
        "messages": [{"role": "user", "content": "hi"}],
        "plugins": [{"id": "x"}],
        "transforms": ["y"],
        "unknown": 123,
    }
    sanitized = policy.sanitize_body(body, endpoint="chat")
    assert "plugins" not in sanitized
    assert "transforms" not in sanitized
    assert "unknown" not in sanitized
    assert "model" in sanitized
    assert "messages" in sanitized


def test_sanitize_embedding_keeps_known_fields(
    proxy_modules: dict[str, object],
) -> None:
    policy = proxy_modules["policy"]
    body = {
        "model": "qwen/qwen3-embedding-8b",
        "input": "hello",
        "dimensions": 512,
        "foo": "bar",
    }
    sanitized = policy.sanitize_body(body, endpoint="embedding")
    assert sanitized == {
        "model": "qwen/qwen3-embedding-8b",
        "input": "hello",
        "dimensions": 512,
    }
