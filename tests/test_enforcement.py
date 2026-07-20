from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest
import respx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
WORKSPACE_URL = "https://openrouter.ai/api/v1/workspaces/workspace-test"


def _chat_response(
    model: str = "deepseek/deepseek-r1-0528",
    *,
    response_id: str = "chatcmpl-test",
    usage: dict[str, int | float] | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": response_id,
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": usage
            or {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


@pytest.mark.asyncio
async def test_unrestricted_model_is_forwarded(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(
            return_value=_chat_response("gpt-4o", response_id="chatcmpl-unrestricted")
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 200
    assert json.loads(route.calls[0].request.content)["model"] == "gpt-4o"


@pytest.mark.asyncio
async def test_restricted_model_is_rejected(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
) -> None:
    proxy_modules["main"].RESTRICTED_MODELS = frozenset({"gpt-4o"})
    response = await client.post(
        "/api/v1/chat/completions",
        headers=auth_headers,
        json={"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 403
    assert "Restricted model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_restricted_model_in_models_array_is_rejected(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
) -> None:
    proxy_modules["main"].RESTRICTED_MODELS = frozenset({"gpt-4o"})
    response = await client.post(
        "/api/v1/chat/completions",
        headers=auth_headers,
        json={
            "model": "deepseek/deepseek-r1-0528",
            "models": ["deepseek/deepseek-r1-0528", "gpt-4o"],
            "messages": [{"role": "user", "content": "hi"}],
        },
    )
    assert response.status_code == 403
    assert "Restricted model" in response.json()["detail"]


@pytest.mark.asyncio
async def test_provider_and_body_fields_are_enforced(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(
            return_value=_chat_response(response_id="chatcmpl-enforcement")
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers={**auth_headers, "X-Custom": "evil"},
            json={
                "model": "deepseek/deepseek-r1-0528",
                "messages": [{"role": "user", "content": "hi"}],
                "provider": {"order": ["evil-provider"]},
                "plugins": [{"id": "bad"}],
                "transforms": ["bad"],
            },
        )

    assert response.status_code == 200
    upstream_request = route.calls[0].request
    upstream_json = json.loads(upstream_request.content)
    provider = upstream_json["provider"]
    # Provider override from request is replaced by the enforced DEFAULT_PROVIDER.
    # DEFAULT_PROVIDER always has allow_fallbacks + zdr; order is optional.
    assert provider.get("allow_fallbacks") is True
    assert provider.get("zdr") is True
    assert provider.get("order") != ["evil-provider"]
    assert upstream_json["usage"] == {"include": True}
    assert "plugins" not in upstream_json
    assert "transforms" not in upstream_json
    assert "x-custom" not in upstream_request.headers
    assert upstream_request.headers["authorization"] == auth_headers["Authorization"]


@pytest.mark.asyncio
async def test_zdr_is_enforced_even_with_per_model_override(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    policy = proxy_modules["policy"]
    monkeypatch.setitem(
        policy.MODEL_PROVIDERS,
        "deepseek/deepseek-r1-0528",
        {"order": ["SomeOther"], "allow_fallbacks": False},
    )

    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(
            return_value=_chat_response(response_id="chatcmpl-zdr")
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek/deepseek-r1-0528",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 200
    upstream_provider = json.loads(route.calls[0].request.content)["provider"]
    assert upstream_provider == {
        "order": ["SomeOther"],
        "allow_fallbacks": False,
        "zdr": True,
    }


@pytest.mark.asyncio
async def test_budget_exceeded_returns_429(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    main = proxy_modules["main"]
    main.cost_tracker.max_cost_usd = 1e-10

    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(
            return_value=_chat_response(
                response_id="chatcmpl-budget",
                usage={
                    "prompt_tokens": 1000,
                    "completion_tokens": 1000,
                    "total_tokens": 2000,
                    "cost": 0.001,
                },
            )
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek/deepseek-r1-0528",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 429


@pytest.mark.asyncio
async def test_runtime_key_mismatch_returns_403(client: httpx.AsyncClient) -> None:
    response = await client.post(
        "/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer sk-wrong-key",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek/deepseek-r1-0528",
            "messages": [{"role": "user", "content": "hi"}],
        },
    )

    assert response.status_code == 403
    assert "does not match the uploaded key" in response.json()["detail"]


@pytest.mark.asyncio
async def test_unsafe_workspace_flags_return_403(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKSPACE_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "data": {
                        "is_observability_io_logging_enabled": True,
                        "is_observability_broadcast_enabled": False,
                        "is_data_discount_logging_enabled": False,
                    }
                },
            )
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek/deepseek-r1-0528",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 403
    assert "logging or data-sharing settings are enabled" in response.json()["detail"]


@pytest.mark.asyncio
async def test_workspace_policy_lookup_failure_returns_503(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=True) as router:
        router.get(WORKSPACE_URL).mock(
            return_value=httpx.Response(
                503, json={"error": {"message": "upstream unavailable"}}
            )
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={
                "model": "deepseek/deepseek-r1-0528",
                "messages": [{"role": "user", "content": "hi"}],
            },
        )

    assert response.status_code == 503
    assert "workspace policy check failed" in response.json()["detail"]
