from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

from .conftest import (
    BYOK_URL,
    KEY_URL,
    WORKSPACE_URL,
    chat_response,
    mock_policy,
)

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
CHAT_BODY = {
    "model": "anthropic/claude-haiku-5.5",
    "messages": [{"role": "user", "content": "hi"}],
}


def _byok_chat_response() -> httpx.Response:
    return chat_response(
        "anthropic/claude-haiku-5.5",
        usage={
            "prompt_tokens": 10,
            "completion_tokens": 1,
            "total_tokens": 11,
            "cost": 0,
            "is_byok": True,
            "cost_details": {"upstream_inference_cost": 0.2},
        },
    )


@pytest.mark.asyncio
async def test_configured_byok_credential_returns_403(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    disabled_credential = {"id": "cred-1", "provider": "anthropic", "disabled": True}
    with respx.mock(assert_all_called=False) as router:
        mock_policy(
            router,
            byok=httpx.Response(
                200, json={"data": [disabled_credential], "total_count": 1}
            ),
        )
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 403
    assert "BYOK provider keys are configured" in response.json()["detail"]
    assert not chat.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body"),
    [
        (500, {"error": {"message": "boom"}}),
        (403, {"error": {"message": "Only management keys can perform this operation"}}),
        (200, {"data": []}),
        (200, {"data": None, "total_count": 0}),
        (200, {"data": [], "total_count": True}),
        (200, ["not", "an", "object"]),
    ],
)
async def test_byok_check_fails_closed(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    status: int,
    body: object,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_policy(router, byok=httpx.Response(status, json=body))
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 503
    assert "BYOK policy check" in response.json()["detail"]
    assert not chat.called


@pytest.mark.asyncio
async def test_workspace_body_without_data_fails_closed(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_policy(router, workspace=httpx.Response(200, json={}))
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 503
    assert not chat.called


@pytest.mark.asyncio
async def test_byok_response_is_withheld_charged_and_blocks_the_run(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    main = proxy_modules["main"]
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        chat = router.post(CHAT_URL).mock(return_value=_byok_chat_response())
        first = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )
        second = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert first.status_code == 403
    assert "choices" not in first.json()
    assert second.status_code == 403
    assert chat.call_count == 1
    assert main.cost_tracker.total_cost() == pytest.approx(0.2)
    assert (tmp_path / main.BYOK_BLOCK_FILENAME).exists()

    usage = json.loads((tmp_path / main.USAGE_FILENAME).read_text())
    assert usage["total_cost_usd"] == pytest.approx(0.2)


@pytest.mark.asyncio
async def test_block_marker_from_sibling_process_blocks_requests(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    main = proxy_modules["main"]
    (tmp_path / main.BYOK_BLOCK_FILENAME).write_text("{}")

    with respx.mock(assert_all_called=False) as router:
        workspace = mock_policy(router)
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 403
    assert not workspace.called
    assert not chat.called


def test_byok_cost_adds_upstream_cost_and_ignores_bad_values(
    proxy_modules: dict[str, object],
) -> None:
    main = proxy_modules["main"]

    assert main._byok_cost(
        {"usage": {"cost": 0.01, "cost_details": {"upstream_inference_cost": 0.2}}}
    ) == pytest.approx(0.21)
    assert main._byok_cost(
        {"usage": {"cost": "nan", "cost_details": {"upstream_inference_cost": "x"}}}
    ) == 0.0
    assert main._byok_cost({"usage": None}) == 0.0


@pytest.mark.asyncio
async def test_key_moved_to_another_workspace_returns_403(
    client: httpx.AsyncClient, auth_headers: dict[str, str]
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_policy(
            router,
            key=httpx.Response(200, json={"data": {"workspace_id": "workspace-other"}}),
        )
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 403
    assert "not in the uploaded workspace" in response.json()["detail"]
    assert not chat.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key_response",
    [
        httpx.Response(404, json={"error": {"message": "not found"}}),
        httpx.Response(200, json={"data": {}}),
        httpx.Response(200, json={"data": {"workspace_id": None}}),
    ],
)
async def test_key_lookup_fails_closed(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    key_response: httpx.Response,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_policy(router, key=key_response)
        chat = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )

    assert response.status_code == 503
    assert "key policy check" in response.json()["detail"]
    assert not chat.called


@pytest.mark.asyncio
async def test_inference_is_sent_only_after_all_three_checks(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, json=CHAT_BODY
        )
        paths = [call.request.url.path for call in router.calls]

    assert response.status_code == 200
    assert paths[-1] == "/api/v1/chat/completions"
    assert set(paths[:-1]) == {
        httpx.URL(KEY_URL).path,
        httpx.URL(WORKSPACE_URL).path,
        httpx.URL(BYOK_URL).path,
    }
