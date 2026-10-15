from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
import respx

from .conftest import chat_response

CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"
EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "deepseek/deepseek-r1-0528"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _payloads(tmp_path: Path) -> list[dict]:
    return _read_jsonl(tmp_path / "proxy_inference_payloads.jsonl")


def _inferences(tmp_path: Path) -> list[dict]:
    return _read_jsonl(tmp_path / "proxy_inferences.jsonl")


def _rebuild(payloads: list[dict]) -> dict[int, dict]:
    """Reference reader: collect every blob map first, then resolve hashes."""
    blobs: dict[str, object] = {}
    for line in payloads:
        blobs.update(line.get("blobs") or {})
    requests = {}
    for line in payloads:
        request = dict(line["request"])
        request["messages"] = [blobs[h] for h in request["messages"]]
        if isinstance(request.get("tools"), str):
            request["tools"] = blobs[request["tools"]]
        requests[line["seq"]] = request
    return requests


async def _chat(
    client: httpx.AsyncClient,
    headers: dict[str, str],
    messages: list[dict],
    **extra: object,
) -> httpx.Response:
    return await client.post(
        "/api/v1/chat/completions",
        headers=headers,
        json={"model": MODEL, "messages": messages, **extra},
    )


@pytest.mark.asyncio
async def test_successful_chat_records_payload_joined_by_seq(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(
            return_value=chat_response(content="hello", response_id="gen-1")
        )
        response = await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    assert response.status_code == 200
    (stats,) = _inferences(tmp_path)
    (payload,) = _payloads(tmp_path)
    assert stats["seq"] == payload["seq"] == 1
    assert payload["status_code"] == 200
    assert "returned_status_code" not in payload
    assert payload["generation_id"] == "gen-1"
    assert payload["request"]["model"] == MODEL
    assert payload["request"]["usage"] == {"include": True}
    assert payload["response"]["choices"][0]["message"]["content"] == "hello"
    assert _rebuild([payload])[1]["messages"] == [{"role": "user", "content": "hi"}]
    assert "sk-fake-test-key" not in (
        tmp_path / "proxy_inference_payloads.jsonl"
    ).read_text()


@pytest.mark.asyncio
async def test_shared_history_is_stored_once(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    first = [{"role": "user", "content": "x" * 1000}]
    second = [
        *first,
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "more"},
    ]
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, first)
        await _chat(client, auth_headers, second)

    payloads = _payloads(tmp_path)
    assert [p["seq"] for p in payloads] == [1, 2]
    assert len(payloads[0]["blobs"]) == 1
    assert len(payloads[1]["blobs"]) == 2  # only the two new messages
    rebuilt = _rebuild(payloads)
    assert rebuilt[1]["messages"] == first
    assert rebuilt[2]["messages"] == second


@pytest.mark.asyncio
async def test_tools_are_stored_once(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    tools = [{"type": "function", "function": {"name": "run", "parameters": {}}}]
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}], tools=tools)
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}], tools=tools)

    first, second = _payloads(tmp_path)
    assert isinstance(first["request"]["tools"], str)
    assert first["request"]["tools"] == second["request"]["tools"]
    assert len(first["blobs"]) == 2  # the message and the tools blob
    assert second["blobs"] == {}
    assert _rebuild([first, second])[2]["tools"] == tools


@pytest.mark.asyncio
async def test_out_of_order_completion_needs_collect_then_resolve(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    shared = {"role": "system", "content": "shared"}

    async def respond(request: httpx.Request) -> httpx.Response:
        if b"slow" in request.content:
            await asyncio.sleep(0.1)
        return chat_response()

    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(side_effect=respond)
        slow = asyncio.create_task(
            _chat(client, auth_headers, [shared, {"role": "user", "content": "slow"}])
        )
        await asyncio.sleep(0.02)
        fast = asyncio.create_task(
            _chat(client, auth_headers, [shared, {"role": "user", "content": "fast"}])
        )
        await asyncio.gather(slow, fast)

    payloads = _payloads(tmp_path)
    assert [p["seq"] for p in payloads] == [2, 1]  # file order != seq order
    # seq=1 references the shared message but seq=2 (earlier in the file) defined it.
    assert len(payloads[1]["blobs"]) == 1
    rebuilt = _rebuild(payloads)
    assert rebuilt[1]["messages"][0] == shared
    assert rebuilt[2]["messages"][0] == shared


@pytest.mark.asyncio
async def test_policy_rejection_records_raw_request_without_response(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    tmp_path: Path,
) -> None:
    headers = {**auth_headers, "Authorization": "Bearer wrong-key"}
    with respx.mock() as router:
        response = await _chat(client, headers, [{"role": "user", "content": "hi"}])
        assert not router.calls

    assert response.status_code == 403
    (payload,) = _payloads(tmp_path)
    assert payload["status_code"] == 403
    assert payload["response"] is None
    assert "does not match" in payload["error"]
    assert _rebuild([payload])[1]["messages"] == [{"role": "user", "content": "hi"}]
    assert "wrong-key" not in (tmp_path / "proxy_inference_payloads.jsonl").read_text()


@pytest.mark.asyncio
async def test_upstream_error_body_is_logged_in_full(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    long_error = {"error": {"message": "boom " * 100}}
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=httpx.Response(502, json=long_error))
        response = await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502
    (payload,) = _payloads(tmp_path)
    assert payload["response"] == long_error
    assert payload["status_code"] == 502


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "upstream",
    [
        pytest.param({"side_effect": httpx.ConnectTimeout("boom")}, id="transport"),
        pytest.param(
            {"return_value": httpx.Response(200, text="not json")}, id="non-json"
        ),
    ],
)
async def test_upstream_failure_is_logged_and_still_raises(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
    upstream: dict,
) -> None:
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(**upstream)
        with pytest.raises(Exception):
            await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    (stats,) = _inferences(tmp_path)
    (payload,) = _payloads(tmp_path)
    assert stats["seq"] == payload["seq"] == 1
    assert stats["status_code"] == payload["status_code"] == 502
    assert payload["response"] is None
    assert payload["error"]
    assert _rebuild([payload])[1]["messages"] == [{"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_budget_override_is_recorded_as_returned_status(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"].cost_tracker.max_cost_usd = 1e-10
    usage = {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.001}
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response(usage=usage))
        response = await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    assert response.status_code == 429
    (payload,) = _payloads(tmp_path)
    assert payload["status_code"] == 200
    assert payload["returned_status_code"] == 429
    assert payload["response"]["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_embedding_vectors_are_replaced_with_dims(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(EMBEDDING_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "emb-1",
                    "model": "openai/text-embedding-3-small",
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "usage": {"prompt_tokens": 2, "cost": 0.0},
                },
            )
        )
        response = await client.post(
            "/api/v1/embeddings",
            headers=auth_headers,
            json={"model": "openai/text-embedding-3-small", "input": "hello"},
        )

    assert response.status_code == 200
    (payload,) = _payloads(tmp_path)
    assert payload["request"]["input"] == "hello"
    assert payload["response"]["data"][0]["embedding"] == {"dims": 3}
    # The agent still receives the real vector.
    assert response.json()["data"][0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_payload_cap_truncates_and_does_not_mark_hashes_seen(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    main = proxy_modules["main"]
    main.MAX_PAYLOAD_LOG_BYTES = 1  # first line fits (check is pre-write), rest truncate
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}])
        await _chat(client, auth_headers, [{"role": "user", "content": "b"}])

    first, second = _payloads(tmp_path)
    assert "truncated" not in first
    assert second["truncated"] is True
    assert second["request"] is None and second["response"] is None
    assert [p["seq"] for p in (first, second)] == [1, 2]
    assert len(main._seen_blob_hashes) == 1  # "b" was never written


@pytest.mark.asyncio
async def test_payload_logging_failure_does_not_break_proxying(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    main = proxy_modules["main"]
    main.PROXY_DATA_DIR = str(tmp_path / "missing" / "dir")
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response(content="fine"))
        response = await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "fine"
    assert len(main._seen_blob_hashes) == 0
