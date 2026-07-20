from __future__ import annotations

import hashlib
import importlib
import sys
from collections.abc import Callable

import httpx
import pytest
import respx

RUNTIME_API_KEY = "sk-fake-test-key"
WORKSPACE_URL = "https://openrouter.ai/api/v1/workspaces/workspace-test"
SAFE_WORKSPACE_BODY = {
    "data": {
        "is_observability_io_logging_enabled": False,
        "is_observability_broadcast_enabled": False,
        "is_data_discount_logging_enabled": False,
    }
}


@pytest.fixture
def proxy_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    monkeypatch.setenv("MAX_COST_USD", "1.0")
    monkeypatch.setenv("EVALUATION_RUN_ID", "eval-run-test")
    monkeypatch.setenv("OPENROUTER_MANAGEMENT_KEY", "sk-or-mgmt-test-key")
    monkeypatch.setenv("OPENROUTER_WORKSPACE_ID", "workspace-test")
    monkeypatch.setenv(
        "OPENROUTER_EXPECTED_API_KEY_SHA256",
        hashlib.sha256(RUNTIME_API_KEY.encode("utf-8")).hexdigest(),
    )
    monkeypatch.delenv("UPSTREAM_BASE_URL", raising=False)

    for name in ("main", "policy", "cost_tracker"):
        sys.modules.pop(name, None)

    policy = importlib.import_module("policy")
    cost_tracker = importlib.import_module("cost_tracker")
    main = importlib.import_module("main")
    return {"policy": policy, "cost_tracker": cost_tracker, "main": main}


@pytest.fixture
async def client(proxy_modules: dict[str, object]) -> httpx.AsyncClient:
    main = proxy_modules["main"]
    app = main.app
    async with httpx.AsyncClient() as upstream:
        app.state.upstream = upstream
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as async_client:
            yield async_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {RUNTIME_API_KEY}",
        "Content-Type": "application/json",
    }


@pytest.fixture
def mock_safe_workspace() -> Callable[[respx.Router], respx.Route]:
    def _mock(router: respx.Router) -> respx.Route:
        return router.get(WORKSPACE_URL).mock(
            return_value=httpx.Response(200, json=SAFE_WORKSPACE_BODY)
        )

    return _mock
