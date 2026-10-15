from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import logging
import os
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, NoReturn

import httpx
from cost_tracker import CostTracker
from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from policy import (
    RESTRICTED_MODELS,
    PolicyError,
    enforce_provider,
    sanitize_body,
    validate_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("ridges.proxy")


def _env_float(name: str, default: float) -> float:
    raw_value = (os.getenv(name) or "").strip()
    if not raw_value:
        return default
    try:
        return float(raw_value)
    except ValueError:
        return default


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw_value = (os.getenv(name) or "").strip()
    if not raw_value:
        return default
    try:
        return max(int(raw_value), minimum)
    except ValueError:
        return default


MAX_INFERENCE_SEED = 2**31 - 1


def _env_optional_seed(name: str) -> int | None:
    raw_value = (os.getenv(name) or "").strip()
    if not raw_value:
        return None
    try:
        value = int(raw_value)
    except ValueError:
        return None
    if value < 0 or value > MAX_INFERENCE_SEED:
        return None
    return value


UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai").rstrip("/")
UPSTREAM_CHAT_PATH = "/api/v1/chat/completions"
UPSTREAM_EMBEDDING_PATH = "/api/v1/embeddings"

EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID", "unknown-eval-run")
MAX_COST_USD = _env_float("MAX_COST_USD", 9999.0)
INFERENCE_SEED = _env_optional_seed("INFERENCE_SEED")
PROXY_DATA_DIR = os.getenv("PROXY_DATA_DIR", "/proxy-data")
MAX_PAYLOAD_LOG_BYTES = _env_int("MAX_PAYLOAD_LOG_BYTES", 64 * 1024 * 1024)
OPENROUTER_MANAGEMENT_KEY = (os.getenv("OPENROUTER_MANAGEMENT_KEY") or "").strip()
OPENROUTER_WORKSPACE_ID = (os.getenv("OPENROUTER_WORKSPACE_ID") or "").strip()
OPENROUTER_EXPECTED_API_KEY_SHA256 = (
    os.getenv("OPENROUTER_EXPECTED_API_KEY_SHA256") or ""
).strip()

cost_tracker = CostTracker(
    evaluation_run_id=EVALUATION_RUN_ID, max_cost_usd=MAX_COST_USD
)


# Persistence helpers
def _flush_usage_to_file() -> None:
    """Atomically write the current cost snapshot to proxy_usage.json (best-effort)."""
    if not PROXY_DATA_DIR:
        return
    try:
        path = os.path.join(PROXY_DATA_DIR, "proxy_usage.json")
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(cost_tracker.dump(), f)
        os.replace(tmp, path)
    except Exception:
        pass


def _append_line(filename: str, line: str) -> bool:
    """Append one line to a file in PROXY_DATA_DIR (best-effort).

    The line goes out in a single write.  If the file does not end in a newline
    (an earlier write was cut short), a newline is written first so the new line
    cannot be glued onto the fragment.
    """
    if not PROXY_DATA_DIR:
        return False
    try:
        fd = os.open(
            os.path.join(PROXY_DATA_DIR, filename),
            os.O_RDWR | os.O_APPEND | os.O_CREAT,
            0o644,
        )
        try:
            size = os.fstat(fd).st_size
            prefix = b"\n" if size and os.pread(fd, 1, size - 1) != b"\n" else b""
            os.write(fd, prefix + line.encode("utf-8") + b"\n")
        finally:
            os.close(fd)
        return True
    except Exception:
        return False


def _append_inference_log(entry: dict[str, Any]) -> None:
    """Append one JSON line to proxy_inferences.jsonl (best-effort)."""
    _append_line("proxy_inferences.jsonl", json.dumps(entry))


# Inference log entry
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


@dataclass
class InferenceLogEntry:
    endpoint: str
    model: str
    status_code: int
    total_cost_usd: float
    budget_remaining_usd: float
    timestamp: str = field(default_factory=_now_iso)
    seq: str | None = None
    routed_model: str | None = None
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    latency_ms: int | None = None
    generation_id: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Payload log (request/response content, joined to the inference log by `seq`)
PAYLOAD_LOG_FILENAME = "proxy_inference_payloads.jsonl"

# `seq` is "<boot id>-<n>": the counter is per process, so the boot id keeps
# the two uvicorn processes (and restarts) from reusing each other's values.
_BOOT_ID = uuid.uuid4().hex[:8]
_seq_counter = itertools.count(1)

_SEEN_BLOB_LIMIT = 200_000
_seen_blob_hashes: OrderedDict[str, None] = OrderedDict()
_log_lock = threading.Lock()


def _next_seq() -> str:
    return f"{_BOOT_ID}-{next(_seq_counter)}"


def _blob_hash(blob: Any) -> str:
    canonical = json.dumps(blob, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=16).hexdigest()


def _mark_blobs_seen(hashes: Any) -> None:
    for digest in hashes:
        _seen_blob_hashes[digest] = None
        _seen_blob_hashes.move_to_end(digest)
    while len(_seen_blob_hashes) > _SEEN_BLOB_LIMIT:
        _seen_blob_hashes.popitem(last=False)


def _dedup_request(request: Any) -> tuple[Any, dict[str, Any]]:
    """Replace `messages` (per message) and `tools` (as one blob) with content hashes.

    Returns (request, blobs first seen by this call).  Chat agents resend the
    whole conversation and the same tool schemas every turn, so storing them
    verbatim would grow quadratically.

    Hashes are global content addresses: a line defines a blob the first time
    this process mentions it, and requests finish out of `seq` order.  Readers
    must therefore collect every `blobs` map in the file before resolving
    hashes, not resolve line by line.  A blob can be defined more than once
    (after a restart, from the other proxy process, or after eviction from the
    bounded seen-set); the content is always identical.
    """
    if not isinstance(request, dict):
        return request, {}
    deduped = dict(request)
    blobs: dict[str, Any] = {}

    def intern(blob: Any) -> str:
        digest = _blob_hash(blob)
        if digest not in _seen_blob_hashes:
            blobs[digest] = blob
        return digest

    if isinstance(request.get("messages"), list):
        deduped["messages"] = [intern(message) for message in request["messages"]]
    if isinstance(request.get("tools"), list):
        deduped["tools"] = intern(request["tools"])
    return deduped, blobs


def _compact_response(endpoint: str, response: Any) -> Any:
    """Replace embedding vectors with their size."""
    if endpoint != "embedding" or not isinstance(response, dict):
        return response
    data = response.get("data")
    if not isinstance(data, list):
        return response

    def compact_item(item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        embedding = item.get("embedding")
        if isinstance(embedding, list):
            return {**item, "embedding": {"dims": len(embedding)}}
        if isinstance(embedding, str):  # encoding_format="base64"
            return {**item, "embedding": {"base64_chars": len(embedding)}}
        return item

    return {**response, "data": [compact_item(item) for item in data]}


@dataclass
class _Exchange:
    """One proxied request, carried through every exit path for logging."""

    seq: str
    endpoint: str
    model: str
    request: Any
    start: float | None = None


def _write_exchange_logs(
    ex: _Exchange,
    *,
    status_code: int,
    error: str | None,
    response: Any,
    returned_status_code: int | None,
    stats: dict[str, Any],
) -> None:
    """Append the stats line and the payload line for one exchange (best-effort).

    `status_code` is the upstream status; `returned_status_code` is set only
    when the agent was sent a different one.  Blocking, so callers run it in a
    thread.  The lock keeps "first seen" in line with file order for the blobs
    this process defines.  Once the payload file reaches MAX_PAYLOAD_LOG_BYTES,
    lines carry no content (`truncated`); the size is checked before each write,
    so one oversized line can overshoot the cap.  Hashes are only marked seen
    after a successful write.
    """
    with _log_lock:
        try:
            _append_inference_log(
                InferenceLogEntry(
                    endpoint=ex.endpoint,
                    model=ex.model,
                    status_code=status_code,
                    total_cost_usd=cost_tracker.total_cost(),
                    budget_remaining_usd=cost_tracker.budget_remaining(),
                    seq=ex.seq,
                    latency_ms=(
                        int((time.monotonic() - ex.start) * 1000)
                        if ex.start is not None
                        else None
                    ),
                    error=error[:200] if error else None,
                    **stats,
                ).to_dict()
            )
        except Exception:
            logger.warning("failed to record stats for seq=%s", ex.seq, exc_info=True)

        if not PROXY_DATA_DIR:
            return
        try:
            entry: dict[str, Any] = {
                "seq": ex.seq,
                "timestamp": _now_iso(),
                "endpoint": ex.endpoint,
                "status_code": status_code,
                "generation_id": stats.get("generation_id"),
                "error": error,
            }
            if returned_status_code is not None:
                entry["returned_status_code"] = returned_status_code
            new_blobs: dict[str, Any] = {}
            try:
                size = os.path.getsize(os.path.join(PROXY_DATA_DIR, PAYLOAD_LOG_FILENAME))
            except OSError:
                size = 0
            if size >= MAX_PAYLOAD_LOG_BYTES:
                entry.update(request=None, response=None, truncated=True)
            else:
                deduped, new_blobs = _dedup_request(ex.request)
                entry.update(
                    request=deduped,
                    blobs=new_blobs,
                    response=_compact_response(ex.endpoint, response),
                )
            if _append_line(PAYLOAD_LOG_FILENAME, json.dumps(entry)):
                _mark_blobs_seen(new_blobs)
        except Exception:
            logger.warning("failed to record payload for seq=%s", ex.seq, exc_info=True)


async def _log_exchange(
    ex: _Exchange,
    *,
    status_code: int,
    error: str | None = None,
    response: Any = None,
    returned_status_code: int | None = None,
    **stats: Any,
) -> None:
    """Log one exchange to both files without blocking the event loop."""
    await asyncio.to_thread(
        _write_exchange_logs,
        ex,
        status_code=status_code,
        error=error,
        response=response,
        returned_status_code=returned_status_code,
        stats=stats,
    )


# Lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    app.state.upstream = httpx.AsyncClient(timeout=300)
    restricted_models = (
        sorted(RESTRICTED_MODELS)
        if RESTRICTED_MODELS
        else "none (all OpenRouter models allowed)"
    )
    logger.info(
        f"proxy startup eval={EVALUATION_RUN_ID} "
        f"max_cost=${MAX_COST_USD:.4f} "
        f"inference_seed={INFERENCE_SEED if INFERENCE_SEED is not None else 'unset'} "
        f"restricted_models={restricted_models}"
    )
    yield
    await app.state.upstream.aclose()
    _flush_usage_to_file()
    logger.info("proxy shutdown %s", cost_tracker.dump())


app = FastAPI(title="Ridges OpenRouter Enforcement Proxy", lifespan=lifespan)


# Request helpers
def _forward_headers(authorization: str, content_type: str | None) -> dict[str, str]:
    headers = {"Authorization": authorization}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _extract_usage(body: dict[str, Any]) -> tuple[int, int, float]:
    usage = body.get("usage", {})
    if not isinstance(usage, dict):
        return 0, 0, 0.0
    return (
        int(usage.get("prompt_tokens", 0) or 0),
        int(usage.get("completion_tokens", 0) or 0),
        float(usage.get("cost", 0.0) or 0.0),
    )


def _extract_provider(body: dict[str, Any]) -> str:
    """Best-effort provider extraction from an OpenRouter response.

    OpenRouter returns the routed model (e.g. 'Chutes/qwen3-coder') in the
    top-level `model` field; the provider is the prefix before the first slash.
    """
    routed: str = body.get("model", "") or ""
    if "/" in routed:
        return routed.split("/")[0]
    return body.get("provider", "") or "unknown"


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    token = parts[1].strip()
    return token or None


def _sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _prepare_payload(raw: dict[str, Any], endpoint: str) -> dict[str, Any]:
    sanitized = sanitize_body(raw, endpoint=endpoint)
    validate_model(sanitized, RESTRICTED_MODELS)
    model = str(sanitized.get("model", ""))
    rewritten = enforce_provider(sanitized, model)
    if endpoint == "chat":
        rewritten["stream"] = False
        rewritten["usage"] = {"include": True}
        if INFERENCE_SEED is not None:
            rewritten["seed"] = INFERENCE_SEED
    return rewritten


async def _reject(ex: _Exchange, status: int, detail: str) -> NoReturn:
    await _log_exchange(ex, status_code=status, error=detail)
    raise HTTPException(status_code=status, detail=detail)


async def _enforce_runtime_openrouter_policy(
    *, authorization: str, ex: _Exchange
) -> None:
    """Fail unless the runtime key matches and the workspace is privacy-safe.

    Every proxied inference must use the uploaded OpenRouter runtime key. This
    helper also checks the owning workspace via the management key and rejects
    requests whenever OpenRouter observability or data-sharing flags are
    enabled.
    """
    token = _extract_bearer_token(authorization)
    if token is None:
        await _reject(ex, 401, "Missing Authorization header")

    if (
        not OPENROUTER_EXPECTED_API_KEY_SHA256
        or not OPENROUTER_MANAGEMENT_KEY
        or not OPENROUTER_WORKSPACE_ID
    ):
        await _reject(
            ex, 503, "OpenRouter runtime policy enforcement is not configured"
        )

    if _sha256_hex(token) != OPENROUTER_EXPECTED_API_KEY_SHA256:
        await _reject(
            ex, 403, "The runtime OpenRouter API key does not match the uploaded key"
        )

    try:
        workspace_response = await app.state.upstream.get(
            f"{UPSTREAM_BASE_URL}/api/v1/workspaces/{OPENROUTER_WORKSPACE_ID}",
            headers={"Authorization": f"Bearer {OPENROUTER_MANAGEMENT_KEY}"},
        )
    except httpx.HTTPError as exc:
        await _reject(
            ex,
            503,
            f"OpenRouter workspace policy check failed: {type(exc).__name__}: {exc}",
        )

    if workspace_response.status_code >= 400:
        await _reject(
            ex,
            503,
            f"OpenRouter workspace policy check failed with status {workspace_response.status_code}",
        )

    try:
        workspace_data = workspace_response.json().get("data") or {}
    except ValueError as exc:
        await _reject(
            ex,
            503,
            f"OpenRouter workspace policy check returned invalid JSON: {exc}",
        )

    enabled_policy_flags = [
        flag_name
        for flag_name in (
            "is_observability_io_logging_enabled",
            "is_observability_broadcast_enabled",
            "is_data_discount_logging_enabled",
        )
        if bool(workspace_data.get(flag_name))
    ]
    if enabled_policy_flags:
        await _reject(
            ex, 403, "OpenRouter workspace logging or data-sharing settings are enabled"
        )


# Core proxy
async def _proxy_to_openrouter(
    *,
    path: str,
    endpoint: str,
    authorization: str | None,
    content_type: str | None,
    payload: dict[str, Any],
) -> JSONResponse:
    ex = _Exchange(
        seq=_next_seq(),
        endpoint=endpoint,
        model=str(payload.get("model", "")),
        request=payload,
    )

    if not authorization or not authorization.strip():
        await _reject(ex, 401, "Missing Authorization header")

    try:
        prepared = _prepare_payload(payload, endpoint=endpoint)
    except PolicyError as exc:
        await _reject(ex, 403, str(exc))

    ex.request = prepared
    ex.model = str(prepared.get("model", ""))

    if not cost_tracker.within_budget():
        await _reject(ex, 429, "Evaluation run cost budget exceeded")

    await _enforce_runtime_openrouter_policy(authorization=authorization, ex=ex)

    ex.start = time.monotonic()
    try:
        upstream_resp = await app.state.upstream.post(
            f"{UPSTREAM_BASE_URL}{path}",
            json=prepared,
            headers=_forward_headers(
                authorization=authorization, content_type=content_type
            ),
        )
    except httpx.HTTPError as exc:
        detail = f"{type(exc).__name__}: {exc}"
        await _log_exchange(ex, status_code=502, error=detail)
        return JSONResponse(
            status_code=502,
            content={"detail": f"Upstream request failed: {type(exc).__name__}"},
        )

    status = upstream_resp.status_code
    try:
        body = upstream_resp.json()
    except ValueError:
        body = None

    if not isinstance(body, dict):
        # Not a JSON object (e.g. an HTML error page): keep the raw text and
        # return a real 502 so the log matches what the agent sees.
        await _log_exchange(
            ex,
            status_code=status,
            error="Upstream returned a non-object response body",
            response={"raw_text": upstream_resp.text[:4096]},
            returned_status_code=502,
        )
        return JSONResponse(
            status_code=502,
            content={"detail": "Upstream returned an invalid response"},
        )

    if status >= 400:
        await _log_exchange(
            ex,
            status_code=status,
            error=json.dumps(body),
            response=body,
        )
        return JSONResponse(status_code=status, content=body)

    prompt_tokens, completion_tokens, cost = _extract_usage(body)
    generation_id: str | None = body.get("id")
    cost_tracker.add_usage(
        model=ex.model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
    )
    _flush_usage_to_file()
    over_budget = not cost_tracker.within_budget()
    await _log_exchange(
        ex,
        status_code=status,
        response=body,
        returned_status_code=429 if over_budget else None,
        routed_model=body.get("model"),
        provider=_extract_provider(body),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=cost,
        generation_id=generation_id,
    )
    if over_budget:
        return JSONResponse(
            status_code=429,
            content={"detail": "Evaluation run cost budget exceeded"},
        )
    return JSONResponse(status_code=status, content=body)


# Routes
@app.get("/healthz")
async def healthz() -> PlainTextResponse:
    return PlainTextResponse("ok\n")


@app.post(UPSTREAM_CHAT_PATH)
async def chat_completions(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
    content_type: str | None = Header(default=None),
) -> JSONResponse:
    return await _proxy_to_openrouter(
        path=UPSTREAM_CHAT_PATH,
        endpoint="chat",
        authorization=authorization,
        content_type=content_type,
        payload=payload,
    )


@app.post(UPSTREAM_EMBEDDING_PATH)
async def embeddings(
    payload: dict[str, Any],
    authorization: str | None = Header(default=None),
    content_type: str | None = Header(default=None),
) -> JSONResponse:
    return await _proxy_to_openrouter(
        path=UPSTREAM_EMBEDDING_PATH,
        endpoint="embedding",
        authorization=authorization,
        content_type=content_type,
        payload=payload,
    )


@app.get("/api/v1/usage")
async def usage() -> JSONResponse:
    return JSONResponse(content=cost_tracker.dump())


@app.api_route(
    "/{full_path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
)
async def deny_all(request: Request, full_path: str) -> Response:
    raise HTTPException(
        status_code=403, detail=f"Path '{request.url.path}' is not allowed"
    )
