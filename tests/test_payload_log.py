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


def _n(seq: str) -> int:
    """Counter part of a `<boot id>-<n>` seq."""
    return int(seq.rsplit("-", 1)[1])


def _rebuild(payloads: list[dict]) -> dict[str, dict]:
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
    assert stats["seq"] == payload["seq"]
    assert _n(payload["seq"]) == 1
    assert payload["status_code"] == 200
    assert "returned_status_code" not in payload
    assert payload["generation_id"] == "gen-1"
    assert payload["request"]["model"] == MODEL
    assert payload["request"]["usage"] == {"include": True}
    assert payload["response"]["choices"][0]["message"]["content"] == "hello"
    (rebuilt,) = _rebuild([payload]).values()
    assert rebuilt["messages"] == [{"role": "user", "content": "hi"}]
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
    assert [_n(p["seq"]) for p in payloads] == [1, 2]
    assert len(payloads[0]["blobs"]) == 1
    assert len(payloads[1]["blobs"]) == 2  # only the two new messages
    rebuilt = _rebuild(payloads)
    assert rebuilt[payloads[0]["seq"]]["messages"] == first
    assert rebuilt[payloads[1]["seq"]]["messages"] == second


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
    assert _rebuild([first, second])[second["seq"]]["tools"] == tools


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
    assert [_n(p["seq"]) for p in payloads] == [2, 1]  # file order != seq order
    # seq 1 references the shared message but seq 2 (earlier in the file) defined it.
    assert len(payloads[1]["blobs"]) == 1
    rebuilt = _rebuild(payloads)
    assert all(r["messages"][0] == shared for r in rebuilt.values())


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
    (rebuilt,) = _rebuild([payload]).values()
    assert rebuilt["messages"] == [{"role": "user", "content": "hi"}]
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
    assert len(payload["error"]) <= 200  # the body is stored once, in `response`


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream", "raw_text"),
    [
        pytest.param({"side_effect": httpx.ConnectTimeout("boom")}, None, id="transport"),
        pytest.param(
            {"return_value": httpx.Response(200, text="not json")},
            "not json",
            id="non-json",
        ),
        pytest.param(
            {"return_value": httpx.Response(200, json=[1, 2])},
            "[1,2]",
            id="json-list",
        ),
        pytest.param(
            {"return_value": httpx.Response(503, text="<html>bad gateway</html>")},
            "<html>bad gateway</html>",
            id="html-error",
        ),
    ],
)
async def test_upstream_failure_is_logged_and_returns_502(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
    upstream: dict,
    raw_text: str | None,
) -> None:
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(**upstream)
        response = await _chat(client, auth_headers, [{"role": "user", "content": "hi"}])

    assert response.status_code == 502
    (stats,) = _inferences(tmp_path)
    (payload,) = _payloads(tmp_path)
    assert stats["seq"] == payload["seq"]
    assert payload["error"]
    if raw_text is None:  # transport error: nothing came back
        assert stats["status_code"] == payload["status_code"] == 502
        assert payload["response"] is None
    else:
        assert payload["returned_status_code"] == stats["returned_status_code"] == 502
        assert stats["status_code"] == payload["status_code"]
        assert payload["response"] == {"raw_text": raw_text}
    (rebuilt,) = _rebuild([payload]).values()
    assert rebuilt["messages"] == [{"role": "user", "content": "hi"}]


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
    (stats,) = _inferences(tmp_path)
    assert stats["status_code"] == 200
    assert stats["returned_status_code"] == 429
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
    assert [_n(p["seq"]) for p in (first, second)] == [1, 2]
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


@pytest.mark.asyncio
async def test_base64_embeddings_are_compacted(
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
                    "id": "emb-2",
                    "model": "openai/text-embedding-3-small",
                    "data": [{"index": 0, "embedding": "A" * 400}],
                    "usage": {"prompt_tokens": 2, "cost": 0.0},
                },
            )
        )
        await client.post(
            "/api/v1/embeddings",
            headers=auth_headers,
            json={
                "model": "openai/text-embedding-3-small",
                "input": "hello",
                "encoding_format": "base64",
            },
        )

    (payload,) = _payloads(tmp_path)
    assert payload["response"]["data"][0]["embedding"] == {"base64_chars": 400}


@pytest.mark.asyncio
async def test_existing_oversized_file_is_already_truncated(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    """The cap follows the file size, so it survives restarts and other processes."""
    proxy_modules["main"].MAX_PAYLOAD_LOG_BYTES = 10
    (tmp_path / "proxy_inference_payloads.jsonl").write_text('{"old": "line"}\n')
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}])

    _, new = _payloads(tmp_path)
    assert new["truncated"] is True


@pytest.mark.asyncio
async def test_append_after_partial_line_does_not_corrupt_next_line(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    path = tmp_path / "proxy_inference_payloads.jsonl"
    path.write_text('{"seq": "cut-off')  # a write that died mid-line
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}])

    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["status_code"] == 200


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("inf", 64 * 1024 * 1024),
        ("nan", 64 * 1024 * 1024),
        ("lots", 64 * 1024 * 1024),
        ("-5", 0),
        ("1024", 1024),
        ("1_000", 1000),
        ("1e9", 1_000_000_000),
        ("6.4e7", 64_000_000),
    ],
)
def test_max_payload_log_bytes_parsing(
    monkeypatch: pytest.MonkeyPatch,
    proxy_modules: dict[str, object],
    raw: str,
    expected: int,
) -> None:
    import importlib
    import sys

    monkeypatch.setenv("MAX_PAYLOAD_LOG_BYTES", raw)
    sys.modules.pop("main", None)
    assert importlib.import_module("main").MAX_PAYLOAD_LOG_BYTES == expected


def test_seen_blob_set_is_bounded(proxy_modules: dict[str, object]) -> None:
    main = proxy_modules["main"]
    main._SEEN_BLOB_LIMIT = 3
    main._mark_blobs_seen([f"h{i}" for i in range(10)])
    assert list(main._seen_blob_hashes) == ["h7", "h8", "h9"]


def _line_is_strict(line: str) -> dict:
    def reject(constant: str) -> None:
        raise ValueError(f"non-strict JSON constant {constant}")

    return json.loads(line, parse_constant=reject)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "raw_body",
    [
        pytest.param(
            b'{"model": "%s", "messages": [{"role": "user", "content": "cut \\ud83d"}]}'
            % MODEL.encode(),
            id="lone-surrogate",
        ),
        pytest.param(
            b'{"model": "%s", "messages": [], "temperature": NaN}' % MODEL.encode(),
            id="nan",
        ),
        pytest.param(
            b'{"model": "%s", "messages": [], "top_p": Infinity}' % MODEL.encode(),
            id="infinity",
        ),
    ],
)
async def test_unencodable_request_is_a_logged_400(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
    raw_body: bytes,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_safe_workspace(router)
        upstream = router.post(CHAT_URL).mock(return_value=chat_response())
        response = await client.post(
            "/api/v1/chat/completions", headers=auth_headers, content=raw_body
        )
        assert not upstream.called

    assert response.status_code == 400
    (stats,) = _inferences(tmp_path)
    assert stats["status_code"] == 400
    path = tmp_path / "proxy_inference_payloads.jsonl"
    lines = path.read_bytes().decode("utf-8").splitlines()  # valid UTF-8
    assert len(lines) == 1
    payload = _line_is_strict(lines[0])
    assert payload["seq"] == stats["seq"]
    assert payload["status_code"] == 400
    assert "cannot be encoded" in payload["error"]


@pytest.mark.asyncio
async def test_short_writes_are_completed_or_reported(
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import errno
    import os

    main = proxy_modules["main"]
    real_write = os.write
    calls = {"n": 0}

    def half_write(fd, data):
        calls["n"] += 1
        return real_write(fd, bytes(data)[: max(1, len(data) // 2)])

    monkeypatch.setattr(main.os, "write", half_write)
    assert main._append_line("a.jsonl", json.dumps({"k": "v" * 50})) is True
    assert calls["n"] >= 2
    assert json.loads((tmp_path / "a.jsonl").read_text())["k"] == "v" * 50

    def half_then_enospc(fd, data):
        if calls["n"] % 2 == 0:
            calls["n"] += 1
            return real_write(fd, bytes(data)[: len(data) // 2])
        raise OSError(errno.ENOSPC, "No space left on device")

    calls["n"] = 0
    monkeypatch.setattr(main.os, "write", half_then_enospc)
    assert main._append_line("b.jsonl", json.dumps({"k": "v" * 50})) is False


@pytest.mark.asyncio
async def test_failed_payload_write_does_not_mark_hashes_seen(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import errno

    main = proxy_modules["main"]
    real_append = main._append_line

    def fail_payloads(filename: str, line: str) -> bool:
        if filename == main.PAYLOAD_LOG_FILENAME:
            main._report_write(filename, OSError(errno.ENOSPC, "full"))
            return False
        return real_append(filename, line)

    monkeypatch.setattr(main, "_append_line", fail_payloads)
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        response = await _chat(client, auth_headers, [{"role": "user", "content": "a"}])

    assert response.status_code == 200
    assert len(main._seen_blob_hashes) == 0


@pytest.mark.asyncio
async def test_stats_are_taken_when_the_exchange_finishes(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import threading
    import time

    main = proxy_modules["main"]
    gate = threading.Event()
    real_write = main._write_exchange_logs

    def slow_write(*args, **kwargs):
        gate.wait(5)
        return real_write(*args, **kwargs)

    monkeypatch.setattr(main, "_write_exchange_logs", slow_write)
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(
            side_effect=[
                chat_response(usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.2}),
                chat_response(usage={"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.4}),
            ]
        )
        first = asyncio.create_task(_chat(client, auth_headers, [{"role": "user", "content": "a"}]))
        await asyncio.sleep(0.05)
        second = asyncio.create_task(_chat(client, auth_headers, [{"role": "user", "content": "b"}]))
        await asyncio.sleep(0.05)
        time.sleep(0.3)  # the log writes sit queued this long
        gate.set()
        await asyncio.gather(first, second)

    a, b = sorted(_inferences(tmp_path), key=lambda e: _n(e["seq"]))
    assert a["total_cost_usd"] == pytest.approx(0.2)
    assert b["total_cost_usd"] == pytest.approx(0.6)
    assert a["latency_ms"] < 250 and b["latency_ms"] < 250


@pytest.mark.asyncio
async def test_other_exits_are_logged(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    workspace_url = "https://openrouter.ai/api/v1/workspaces/workspace-test"
    with respx.mock(assert_all_called=False) as router:
        router.get(workspace_url).mock(return_value=httpx.Response(200, json=[]))
        upstream = router.post(CHAT_URL).mock(return_value=chat_response())
        r1 = await _chat(client, auth_headers, [{"role": "user", "content": "1"}])
        assert not upstream.called
    assert r1.status_code == 503

    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(
            return_value=chat_response(usage={"prompt_tokens": "n/a"})
        )
        r2 = await _chat(client, auth_headers, [{"role": "user", "content": "2"}])
    assert r2.status_code == 502

    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(
            return_value=httpx.Response(
                200,
                json={
                    "id": "g",
                    "model": 5,
                    "choices": [],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "cost": 0.25},
                },
            )
        )
        r3 = await _chat(client, auth_headers, [{"role": "user", "content": "3"}])
    assert r3.status_code == 200

    s1, s2, s3 = _inferences(tmp_path)
    assert (s1["status_code"], s2["status_code"], s3["status_code"]) == (503, 200, 200)
    assert s2["returned_status_code"] == 502
    assert s3["provider"] == "unknown" and s3["routed_model"] is None
    assert s3["cost_usd"] == 0.25
    assert len(_payloads(tmp_path)) == 3


@pytest.mark.asyncio
async def test_unexpected_error_is_logged_then_propagates(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    main = proxy_modules["main"]

    def boom(*args, **kwargs):
        raise RuntimeError("kaput")

    monkeypatch.setattr(main, "_prepare_payload", boom)
    with pytest.raises(RuntimeError):
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}])

    (stats,) = _inferences(tmp_path)
    assert stats["status_code"] == 500
    assert "kaput" in stats["error"]
    assert len(_payloads(tmp_path)) == 1


@pytest.mark.asyncio
async def test_write_failures_warn_once_and_recovery_is_reported(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    caplog: pytest.LogCaptureFixture,
    tmp_path: Path,
) -> None:
    import logging

    main = proxy_modules["main"]
    missing = tmp_path / "missing"
    main.PROXY_DATA_DIR = str(missing)
    caplog.set_level(logging.INFO, logger="ridges.proxy")
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        for i in range(3):
            await _chat(client, auth_headers, [{"role": "user", "content": str(i)}])
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert sorted(r.args[0] for r in warnings) == [
            "proxy_inference_payloads.jsonl",
            "proxy_inferences.jsonl",
            "proxy_usage.json",
        ]
        missing.mkdir()
        await _chat(client, auth_headers, [{"role": "user", "content": "ok"}])

    assert sum("works again" in r.getMessage() for r in caplog.records) == 3
    assert len(_inferences(missing)) == 1


@pytest.mark.asyncio
async def test_log_writes_run_on_one_thread_and_usage_is_written(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import threading

    main = proxy_modules["main"]
    names: set[str] = set()
    real_append = main._append_line

    def spy(filename: str, line: str) -> bool:
        names.add(threading.current_thread().name)
        return real_append(filename, line)

    monkeypatch.setattr(main, "_append_line", spy)
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": "a"}])
        await _chat(client, auth_headers, [{"role": "user", "content": "b"}])

    assert len(names) == 1 and next(iter(names)).startswith("proxy-log")
    assert json.loads((tmp_path / "proxy_usage.json").read_text())


def test_append_waits_for_the_file_lock(
    proxy_modules: dict[str, object], tmp_path: Path
) -> None:
    import fcntl
    import os
    import threading

    main = proxy_modules["main"]
    path = tmp_path / "locked.jsonl"
    other = os.open(path, os.O_RDWR | os.O_APPEND | os.O_CREAT, 0o644)
    fcntl.flock(other, fcntl.LOCK_EX)
    os.write(other, b'{"seq": 1, "half')  # another process, mid-line
    results: list[bool] = []
    thread = threading.Thread(
        target=lambda: results.append(main._append_line("locked.jsonl", '{"seq": 3}'))
    )
    thread.start()
    thread.join(0.2)
    assert thread.is_alive()  # blocked on the lock

    os.write(other, b'": "line"}\n{"seq": 2}\n')
    fcntl.flock(other, fcntl.LOCK_UN)
    os.close(other)
    thread.join(5)

    assert results == [True]
    lines = path.read_text().splitlines()
    assert [json.loads(line).get("seq") for line in lines] == [1, 2, 3]


@pytest.mark.asyncio
async def test_non_ascii_is_stored_as_utf8_and_stays_one_line_per_record(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    tmp_path: Path,
) -> None:
    text = "日本語のテキスト" * 5 + "a\u2028b\u0085c\u2029d"
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        await _chat(client, auth_headers, [{"role": "user", "content": text}])
        await _chat(client, auth_headers, [{"role": "user", "content": text + "!"}])

    raw = (tmp_path / "proxy_inference_payloads.jsonl").read_text(encoding="utf-8")
    assert "日本語のテキスト" in raw  # not \\u-escaped
    assert len(raw.splitlines()) == 2
    rebuilt = _rebuild([json.loads(line) for line in raw.splitlines()])
    assert {r["messages"][0]["content"] for r in rebuilt.values()} == {text, text + "!"}


@pytest.mark.asyncio
async def test_shared_prompt_stays_defined_once_under_a_small_seen_limit(
    client: httpx.AsyncClient,
    auth_headers: dict[str, str],
    mock_safe_workspace: Callable[[respx.Router], respx.Route],
    proxy_modules: dict[str, object],
    tmp_path: Path,
) -> None:
    proxy_modules["main"]._SEEN_BLOB_LIMIT = 3
    system = {"role": "system", "content": "shared prompt"}
    with respx.mock() as router:
        mock_safe_workspace(router)
        router.post(CHAT_URL).mock(return_value=chat_response())
        for i in range(10):
            await _chat(
                client, auth_headers, [system, {"role": "user", "content": f"q{i}"}]
            )

    definitions = [
        digest
        for payload in _payloads(tmp_path)
        for digest, blob in payload["blobs"].items()
        if blob == system
    ]
    assert len(definitions) == 1
