"""Capture on unusual request and response shapes: nothing may break or change for the agent."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "deepseek/deepseek-r1-0528"
LOGPROBS = {"content": [{"token": "ok", "logprob": -0.1, "bytes": [111, 107]}]}
USAGE = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _choice(index: int, **message: object) -> dict:
    return {
        "index": index,
        "message": {"role": "assistant", "content": f"answer {index}", **message},
        "logprobs": LOGPROBS,
        "finish_reason": "stop",
    }


@pytest.fixture
def capture_on(proxy_modules: dict[str, object]) -> object:
    main = proxy_modules["main"]
    main.CAPTURE_LOGPROBS = True
    main.CAPTURE_REASONING = True
    main._logprob_models.add(MODEL)
    return main


async def _post(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    upstream: httpx.Response,
    **extra: object,
) -> tuple[dict, httpx.Response]:
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(return_value=upstream)
        response = await client.post(
            "/api/v1/chat/completions",
            headers=headers,
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": "hi"}],
                **extra,
            },
        )
    return json.loads(route.calls[0].request.content), response


@pytest.mark.asyncio
async def test_every_choice_is_captured_and_hidden(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    body = {
        "id": "gen-n2",
        "model": MODEL,
        "choices": [_choice(0, reasoning="r0"), _choice(1, reasoning="r1")],
        "usage": USAGE,
    }
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        n=2,
        reasoning={"exclude": True},
    )

    choices = response.json()["choices"]
    assert [c["message"] for c in choices] == [
        {"role": "assistant", "content": "answer 0"},
        {"role": "assistant", "content": "answer 1"},
    ]
    assert [c["logprobs"] for c in choices] == [None, None]
    (captured,) = _read_jsonl(tmp_path / "proxy_logprobs.jsonl")
    assert [c["index"] for c in captured["choices"]] == [0, 1]


@pytest.mark.asyncio
async def test_over_budget_reply_still_records_capture(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    capture_on.cost_tracker.max_cost_usd = 1e-10
    body = {
        "id": "gen-budget",
        "model": MODEL,
        "choices": [_choice(0, reasoning="thought")],
        "usage": {**USAGE, "cost": 0.001},
    }
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        reasoning={"exclude": True},
    )

    assert response.status_code == 429
    (payload,) = _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")
    assert payload["returned_status_code"] == 429
    assert payload["response"]["choices"][0]["message"]["reasoning"] == "thought"
    assert len(_read_jsonl(tmp_path / "proxy_logprobs.jsonl")) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reasoning", ["high", None, True, ["exclude"], {"exclude": 0}, {"effort": "low"}]
)
async def test_unusual_reasoning_values_pass_through(
    reasoning: object,
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    body = {
        "id": "gen-r",
        "model": MODEL,
        "choices": [_choice(0, reasoning="visible")],
        "usage": USAGE,
    }
    sent, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        reasoning=reasoning,
    )

    assert sent["reasoning"] == reasoning
    assert response.json()["choices"][0]["message"]["reasoning"] == "visible"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "choices",
    [
        [],
        ["not-a-dict"],
        [{"index": 0, "finish_reason": "stop"}],
        [{"index": 0, "message": None, "logprobs": None}],
        [{"index": 0, "message": "text", "logprobs": "junk"}],
        None,
    ],
    ids=[
        "empty",
        "string-choice",
        "no-message",
        "null-message",
        "string-message",
        "no-choices",
    ],
)
async def test_odd_response_shapes_reach_agent_unchanged(
    choices: object,
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    body: dict = {"id": "gen-odd", "model": MODEL, "usage": USAGE}
    if choices is not None:
        body["choices"] = choices
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        reasoning={"exclude": True},
    )

    assert response.status_code == 200
    expected = json.loads(json.dumps(body))
    for choice in expected.get("choices") or []:
        if isinstance(choice, dict) and "logprobs" in choice:
            choice["logprobs"] = None  # the only change capture makes here
    assert response.json() == expected
    assert len(_read_jsonl(tmp_path / "proxy_inferences.jsonl")) == 1


@pytest.mark.asyncio
async def test_tool_calls_survive_reasoning_strip(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    tool_calls = [
        {
            "id": "call-1",
            "type": "function",
            "function": {"name": "read_file", "arguments": '{"path": "a.py"}'},
        }
    ]
    body = {
        "id": "gen-tools",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": tool_calls,
                    "reasoning": "need the file",
                    "reasoning_details": [
                        {"type": "reasoning.text", "text": "need the file"}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": USAGE,
    }
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        reasoning={"exclude": True},
    )

    message = response.json()["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "", "tool_calls": tool_calls}
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_upstream_error_reaches_agent_unchanged(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    error = {"error": {"message": "bad request", "code": 400}}
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(400, json=error),
        reasoning={"exclude": True},
    )

    assert response.status_code == 400
    assert response.json() == error
    assert not (tmp_path / "proxy_logprobs.jsonl").exists()


@pytest.mark.asyncio
async def test_embeddings_are_untouched_by_capture(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    model = "openai/text-embedding-3-small"
    capture_on._logprob_models.add(model)
    body = {
        "id": "emb-1",
        "model": model,
        "data": [{"index": 0, "embedding": [0.1, 0.2]}],
        "usage": {"prompt_tokens": 2, "cost": 0.0},
    }
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(EMBEDDING_URL).mock(
            return_value=httpx.Response(200, json=body)
        )
        response = await client.post(
            "/api/v1/embeddings",
            headers=auth_headers,
            json={"model": model, "input": "hello"},
        )

    sent = json.loads(route.calls[0].request.content)
    assert "logprobs" not in sent and "reasoning" not in sent
    assert response.json() == body


def _ok(**extra: object) -> httpx.Response:
    body = {"id": "gen-ok", "model": MODEL, "choices": [_choice(0)], "usage": USAGE}
    return httpx.Response(200, json={**body, **extra})


@pytest.mark.asyncio
async def test_rejected_logprobs_are_dropped_and_resent(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    # Mistral answers this way although the catalog lists logprobs.
    refused = httpx.Response(
        400, json={"error": {"message": "Logprobs are not enabled for this model"}}
    )
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(side_effect=[refused, _ok(), _ok()])
        for _ in range(2):
            response = await client.post(
                "/api/v1/chat/completions",
                headers=auth_headers,
                json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
            )
            assert response.status_code == 200

    sent = [json.loads(call.request.content) for call in route.calls]
    assert [("logprobs" in s) for s in sent] == [True, False, False]
    assert MODEL not in capture_on._logprob_models
    stats = _read_jsonl(tmp_path / "proxy_inferences.jsonl")
    assert [s["status_code"] for s in stats] == [200, 200]
    first, _ = _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")
    assert "logprobs" not in first["request"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [429, 500])
async def test_other_failures_are_not_retried(
    status: int,
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    error = {"error": {"message": "busy"}}
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(
            return_value=httpx.Response(status, json=error)
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == status
    assert len(route.calls) == 1
    assert MODEL in capture_on._logprob_models


@pytest.mark.asyncio
async def test_client_error_without_added_logprobs_is_not_retried(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    capture_on._logprob_models.clear()
    with respx.mock(assert_all_called=True) as router:
        mock_safe_workspace(router)
        route = router.post(CHAT_URL).mock(
            return_value=httpx.Response(400, json={"error": {"message": "bad"}})
        )
        response = await client.post(
            "/api/v1/chat/completions",
            headers=auth_headers,
            json={"model": MODEL, "messages": [{"role": "user", "content": "hi"}]},
        )

    assert response.status_code == 400
    assert len(route.calls) == 1


@pytest.mark.asyncio
async def test_signed_and_encrypted_reasoning_reach_agent(
    capture_on: object,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
) -> None:
    # Shapes seen live with exclude: true. Anthropic returns signed text,
    # Gemini an encrypted entry; open models return plain text only with
    # exclude: false.
    signed = {
        "type": "reasoning.text",
        "text": "thinking",
        "signature": "sig",
        "format": "anthropic-claude-v1",
    }
    encrypted = {
        "type": "reasoning.encrypted",
        "data": "x",
        "format": "google-gemini-v1",
    }
    plain = {"type": "reasoning.text", "text": "thinking", "format": "unknown"}
    summary = {
        "type": "reasoning.summary",
        "summary": "s",
        "format": "openai-responses-v1",
    }
    details = [signed, encrypted, plain, summary]
    body = {
        "id": "gen-sig",
        "model": MODEL,
        "choices": [_choice(0, reasoning="thinking", reasoning_details=details)],
        "usage": USAGE,
    }
    _, response = await _post(
        client,
        auth_headers,
        mock_safe_workspace,
        httpx.Response(200, json=body),
        reasoning={"exclude": True},
    )

    message = response.json()["choices"][0]["message"]
    assert message["reasoning_details"] == [signed, encrypted]
    assert "reasoning" not in message
