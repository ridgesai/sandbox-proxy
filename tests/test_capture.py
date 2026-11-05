from __future__ import annotations

import asyncio
import importlib
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
MODELS_URL = "https://openrouter.ai/api/v1/models"
MODEL = "deepseek/deepseek-r1-0528"
LOGPROBS = {"content": [{"token": "ok", "logprob": -0.1, "bytes": [111, 107]}]}
REASONING = "think first"
REASONING_DETAILS = [{"type": "reasoning.text", "text": REASONING}]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _upstream_response(
    *,
    logprobs: object = None,
    reasoning: bool = False,
    details: list[dict] = REASONING_DETAILS,
) -> httpx.Response:
    message: dict[str, object] = {"role": "assistant", "content": "ok"}
    if reasoning:
        message["reasoning"] = REASONING
        message["reasoning_details"] = details
    return httpx.Response(
        200,
        json={
            "id": "gen-capture",
            "object": "chat.completion",
            "model": MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "logprobs": logprobs,
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    )


async def _exchange(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    upstream: httpx.Response,
    **extra: object,
) -> tuple[dict, dict]:
    """Send one chat call; return (body sent upstream, body the agent got)."""
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
    assert response.status_code == 200
    return json.loads(route.calls[0].request.content), response.json()


@pytest.mark.asyncio
async def test_capture_can_be_disabled(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].CAPTURE_LOGPROBS = False
    proxy_modules["main"].CAPTURE_REASONING = False

    sent, _ = await _exchange(
        client,
        auth_headers,
        mock_safe_workspace,
        _upstream_response(),
        logprobs=True,
        reasoning={"exclude": True},
    )

    assert "logprobs" not in sent  # still stripped by the allow-list
    assert sent["reasoning"] == {"exclude": True}
    assert not (tmp_path / "proxy_logprobs.jsonl").exists()


@pytest.mark.asyncio
async def test_logprobs_are_captured_and_hidden_from_agent(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].CAPTURE_LOGPROBS = True
    proxy_modules["main"]._logprob_models.add(MODEL)
    proxy_modules["main"].CAPTURE_TOP_LOGPROBS = 5

    sent, received = await _exchange(
        client, auth_headers, mock_safe_workspace, _upstream_response(logprobs=LOGPROBS)
    )

    assert sent["logprobs"] is True
    assert sent["top_logprobs"] == 5
    assert received["choices"][0]["logprobs"] is None
    assert received["choices"][0]["message"]["content"] == "ok"
    (payload,) = _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")
    assert payload["response"]["choices"][0]["logprobs"] is None
    (captured,) = _read_jsonl(tmp_path / "proxy_logprobs.jsonl")
    assert captured["seq"] == payload["seq"]
    assert captured["generation_id"] == "gen-capture"
    assert captured["choices"] == [{"index": 0, "logprobs": LOGPROBS}]


@pytest.mark.asyncio
async def test_top_logprobs_is_omitted_when_zero(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
) -> None:
    proxy_modules["main"].CAPTURE_LOGPROBS = True
    proxy_modules["main"]._logprob_models.add(MODEL)

    sent, _ = await _exchange(
        client, auth_headers, mock_safe_workspace, _upstream_response()
    )

    assert sent["logprobs"] is True
    assert "top_logprobs" not in sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extra",
    [{}, {"models": [MODEL, "openai/gpt-6-luna"]}],
    ids=["model-not-in-catalog", "fallback-not-in-catalog"],
)
async def test_logprobs_only_sent_to_catalog_models(
    extra: dict,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
) -> None:
    main = proxy_modules["main"]
    main.CAPTURE_LOGPROBS = True
    if extra:
        main._logprob_models.add(MODEL)  # the fallback is the one missing

    sent, _ = await _exchange(
        client, auth_headers, mock_safe_workspace, _upstream_response(), **extra
    )

    assert "logprobs" not in sent


@pytest.mark.asyncio
async def test_catalog_load_records_logprob_models(
    proxy_modules: dict[str, object],
) -> None:
    main = proxy_modules["main"]
    catalog = {
        "data": [
            {"id": MODEL, "supported_parameters": ["tools", "logprobs"]},
            {"id": "openai/gpt-6-luna", "supported_parameters": ["reasoning"]},
            {"id": "no-params/model"},
        ]
    }
    with respx.mock() as router:
        router.get(MODELS_URL).mock(return_value=httpx.Response(200, json=catalog))
        async with httpx.AsyncClient() as upstream:
            await main._load_logprob_models(upstream)

    assert main._logprob_models == {MODEL}


@pytest.mark.asyncio
async def test_startup_loads_catalog_in_background(
    proxy_modules: dict[str, object],
) -> None:
    main = proxy_modules["main"]
    catalog = {"data": [{"id": MODEL, "supported_parameters": ["logprobs"]}]}
    with respx.mock() as router:
        router.get(MODELS_URL).mock(return_value=httpx.Response(200, json=catalog))
        async with main.lifespan(main.app):
            for _ in range(100):
                if main._logprob_models:
                    break
                await asyncio.sleep(0.01)

    assert main._logprob_models == {MODEL}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(503, text="down"),
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"unexpected": True}),
        httpx.Response(200, json=[{"id": MODEL}]),
        httpx.Response(200, json={"data": 7}),
        httpx.Response(200, json={"data": [{"supported_parameters": ["logprobs"]}]}),
        httpx.Response(
            200, json={"data": [{"id": [MODEL], "supported_parameters": ["logprobs"]}]}
        ),
    ],
    ids=[
        "http-error",
        "bad-json",
        "bad-shape",
        "list-body",
        "data-not-list",
        "entry-without-id",
        "unhashable-id",
    ],
)
async def test_catalog_failure_leaves_logprobs_off(
    reply: httpx.Response, proxy_modules: dict[str, object]
) -> None:
    main = proxy_modules["main"]
    with respx.mock() as router:
        router.get(MODELS_URL).mock(return_value=reply)
        async with httpx.AsyncClient() as upstream:
            await main._load_logprob_models(upstream)

    assert main._logprob_models == set()


@pytest.mark.asyncio
async def test_no_logprobs_line_when_provider_returns_none(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].CAPTURE_LOGPROBS = True
    proxy_modules["main"]._logprob_models.add(MODEL)

    await _exchange(client, auth_headers, mock_safe_workspace, _upstream_response())

    assert not (tmp_path / "proxy_logprobs.jsonl").exists()


@pytest.mark.asyncio
async def test_logprobs_log_stops_at_its_cap(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].CAPTURE_LOGPROBS = True
    proxy_modules["main"]._logprob_models.add(MODEL)
    proxy_modules["main"].MAX_LOGPROBS_LOG_BYTES = 1  # only the first line fits

    for _ in range(2):
        await _exchange(
            client,
            auth_headers,
            mock_safe_workspace,
            _upstream_response(logprobs=LOGPROBS),
        )

    first, second = _read_jsonl(tmp_path / "proxy_logprobs.jsonl")
    assert first["choices"] == [{"index": 0, "logprobs": LOGPROBS}]
    assert second["truncated"] is True
    assert "choices" not in second


@pytest.mark.asyncio
async def test_excluded_reasoning_is_captured_and_hidden_from_agent(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].CAPTURE_REASONING = True

    sent, received = await _exchange(
        client,
        auth_headers,
        mock_safe_workspace,
        _upstream_response(reasoning=True),
        reasoning={"effort": "low", "exclude": True},
    )

    assert sent["reasoning"] == {"effort": "low", "exclude": False}
    message = received["choices"][0]["message"]
    assert message == {"role": "assistant", "content": "ok"}
    (payload,) = _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")
    logged = payload["response"]["choices"][0]["message"]
    assert logged["reasoning"] == REASONING
    assert logged["reasoning_details"] == REASONING_DETAILS
    assert payload["received_request"]["reasoning"] == {
        "effort": "low",
        "exclude": True,
    }


@pytest.mark.asyncio
async def test_encrypted_reasoning_still_reaches_agent(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    # OpenAI models return the encrypted entry even with exclude: true.
    encrypted = {"type": "reasoning.encrypted", "data": "opaque"}
    summary = {"type": "reasoning.summary", "summary": REASONING}
    proxy_modules["main"].CAPTURE_REASONING = True

    _, received = await _exchange(
        client,
        auth_headers,
        mock_safe_workspace,
        _upstream_response(reasoning=True, details=[encrypted, summary]),
        reasoning={"exclude": True},
    )

    message = received["choices"][0]["message"]
    assert message["reasoning_details"] == [encrypted]
    assert "reasoning" not in message
    (payload,) = _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")
    logged = payload["response"]["choices"][0]["message"]
    assert logged["reasoning_details"] == [encrypted, summary]


@pytest.mark.asyncio
@pytest.mark.parametrize("reasoning", [None, {"effort": "low"}, {"exclude": False}])
async def test_reasoning_the_agent_did_not_exclude_is_untouched(
    reasoning: dict | None,
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
) -> None:
    proxy_modules["main"].CAPTURE_REASONING = True
    extra = {} if reasoning is None else {"reasoning": reasoning}

    sent, received = await _exchange(
        client,
        auth_headers,
        mock_safe_workspace,
        _upstream_response(reasoning=True),
        **extra,
    )

    assert sent.get("reasoning") == reasoning
    assert received["choices"][0]["message"]["reasoning"] == REASONING


def test_capture_settings_read_from_env(
    monkeypatch: pytest.MonkeyPatch, proxy_modules: dict[str, object]
) -> None:
    main = proxy_modules["main"]
    assert main.CAPTURE_LOGPROBS is True  # on when unset
    assert main.CAPTURE_TOP_LOGPROBS == 0
    assert main.CAPTURE_REASONING is True

    monkeypatch.setenv("CAPTURE_LOGPROBS", "false")
    monkeypatch.setenv("CAPTURE_TOP_LOGPROBS", "50")
    monkeypatch.setenv("CAPTURE_REASONING", "0")
    main = importlib.reload(main)

    assert main.CAPTURE_LOGPROBS is False
    assert main.CAPTURE_TOP_LOGPROBS == 20  # the API maximum
    assert main.CAPTURE_REASONING is False

    monkeypatch.setenv("CAPTURE_LOGPROBS", "not-a-bool")
    monkeypatch.setenv("CAPTURE_REASONING", "not-a-bool")
    main = importlib.reload(main)
    assert main.CAPTURE_LOGPROBS is True  # junk falls back to the default
    assert main.CAPTURE_REASONING is True
