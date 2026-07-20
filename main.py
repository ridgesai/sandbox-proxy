from __future__ import annotations

import hashlib
import json
import logging
import os
import time
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


UPSTREAM_BASE_URL = os.getenv("UPSTREAM_BASE_URL", "https://openrouter.ai").rstrip("/")
UPSTREAM_CHAT_PATH = "/api/v1/chat/completions"
UPSTREAM_EMBEDDING_PATH = "/api/v1/embeddings"

EVALUATION_RUN_ID = os.getenv("EVALUATION_RUN_ID", "unknown-eval-run")
MAX_COST_USD = _env_float("MAX_COST_USD", 9999.0)
PROXY_DATA_DIR = os.getenv("PROXY_DATA_DIR", "/proxy-data")
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


def _append_inference_log(entry: dict[str, Any]) -> None:
    """Append one JSON line to proxy_inferences.jsonl (best-effort)."""
    if not PROXY_DATA_DIR:
        return
    try:
        with open(os.path.join(PROXY_DATA_DIR, "proxy_inferences.jsonl"), "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


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
    return rewritten


def _reject(
    status: int,
    detail: str,
    *,
    endpoint: str,
    model: str,
) -> NoReturn:
    _append_inference_log(
        InferenceLogEntry(
            endpoint=endpoint,
            model=model,
            status_code=status,
            total_cost_usd=cost_tracker.total_cost(),
            budget_remaining_usd=cost_tracker.budget_remaining(),
            error=detail,
        ).to_dict()
    )
    raise HTTPException(status_code=status, detail=detail)


async def _enforce_runtime_openrouter_policy(
    *, authorization: str, endpoint: str, model: str
) -> None:
    """Fail unless the runtime key matches and the workspace is privacy-safe.

    Every proxied inference must use the uploaded OpenRouter runtime key. This
    helper also checks the owning workspace via the management key and rejects
    requests whenever OpenRouter observability or data-sharing flags are
    enabled.
    """
    token = _extract_bearer_token(authorization)
    if token is None:
        _reject(401, "Missing Authorization header", endpoint=endpoint, model=model)

    if (
        not OPENROUTER_EXPECTED_API_KEY_SHA256
        or not OPENROUTER_MANAGEMENT_KEY
        or not OPENROUTER_WORKSPACE_ID
    ):
        _reject(
            503,
            "OpenRouter runtime policy enforcement is not configured",
            endpoint=endpoint,
            model=model,
        )

    if _sha256_hex(token) != OPENROUTER_EXPECTED_API_KEY_SHA256:
        _reject(
            403,
            "The runtime OpenRouter API key does not match the uploaded key",
            endpoint=endpoint,
            model=model,
        )

    try:
        workspace_response = await app.state.upstream.get(
            f"{UPSTREAM_BASE_URL}/api/v1/workspaces/{OPENROUTER_WORKSPACE_ID}",
            headers={"Authorization": f"Bearer {OPENROUTER_MANAGEMENT_KEY}"},
        )
    except httpx.HTTPError as exc:
        _reject(
            503,
            f"OpenRouter workspace policy check failed: {type(exc).__name__}: {exc}",
            endpoint=endpoint,
            model=model,
        )

    if workspace_response.status_code >= 400:
        _reject(
            503,
            f"OpenRouter workspace policy check failed with status {workspace_response.status_code}",
            endpoint=endpoint,
            model=model,
        )

    try:
        workspace_data = workspace_response.json().get("data") or {}
    except ValueError as exc:
        _reject(
            503,
            f"OpenRouter workspace policy check returned invalid JSON: {exc}",
            endpoint=endpoint,
            model=model,
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
        _reject(
            403,
            "OpenRouter workspace logging or data-sharing settings are enabled",
            endpoint=endpoint,
            model=model,
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
    requested_model = str(payload.get("model", ""))

    if not authorization or not authorization.strip():
        _reject(
            401,
            "Missing Authorization header",
            endpoint=endpoint,
            model=requested_model,
        )

    try:
        prepared = _prepare_payload(payload, endpoint=endpoint)
    except PolicyError as exc:
        _reject(
            403,
            str(exc),
            endpoint=endpoint,
            model=requested_model,
        )

    model = str(prepared.get("model", ""))

    if not cost_tracker.within_budget():
        _reject(
            429, "Evaluation run cost budget exceeded", endpoint=endpoint, model=model
        )

    await _enforce_runtime_openrouter_policy(
        authorization=authorization,
        endpoint=endpoint,
        model=model,
    )

    start = time.monotonic()
    upstream_resp = await app.state.upstream.post(
        f"{UPSTREAM_BASE_URL}{path}",
        json=prepared,
        headers=_forward_headers(
            authorization=authorization, content_type=content_type
        ),
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    body = upstream_resp.json()

    if upstream_resp.status_code < 400:
        prompt_tokens, completion_tokens, cost = _extract_usage(body)
        provider = _extract_provider(body)
        routed_model: str | None = body.get("model")
        generation_id: str | None = body.get("id")
        cost_tracker.add_usage(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )
        _flush_usage_to_file()
        _append_inference_log(
            InferenceLogEntry(
                endpoint=endpoint,
                model=model,
                status_code=upstream_resp.status_code,
                total_cost_usd=cost_tracker.total_cost(),
                budget_remaining_usd=cost_tracker.budget_remaining(),
                routed_model=routed_model,
                provider=provider,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                generation_id=generation_id,
            ).to_dict()
        )
        if not cost_tracker.within_budget():
            return JSONResponse(
                status_code=429,
                content={"detail": "Evaluation run cost budget exceeded"},
            )
    else:
        error_snippet = json.dumps(body)[:200]
        _append_inference_log(
            InferenceLogEntry(
                endpoint=endpoint,
                model=model,
                status_code=upstream_resp.status_code,
                total_cost_usd=cost_tracker.total_cost(),
                budget_remaining_usd=cost_tracker.budget_remaining(),
                latency_ms=latency_ms,
                error=error_snippet,
            ).to_dict()
        )

    return JSONResponse(status_code=upstream_resp.status_code, content=body)


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
