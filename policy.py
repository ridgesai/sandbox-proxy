from __future__ import annotations

import os
from copy import deepcopy
from typing import Any


def _parse_provider_order(raw: str) -> list[str]:
    return [p.strip() for p in raw.split(",") if p.strip()]


_provider_order_raw = os.getenv("PROVIDER_ORDER", "")
_allow_fallbacks = os.getenv("PROVIDER_ALLOW_FALLBACKS", "true").lower() != "false"
_restricted_models_raw = os.getenv("RESTRICTED_MODELS", "")

# Deployment-wide model and provider config.
RESTRICTED_MODELS: frozenset[str] = frozenset(
    m.strip() for m in _restricted_models_raw.split(",") if m.strip()
)

# Global OpenRouter provider preferences applied to every request.
DEFAULT_PROVIDER: dict[str, Any] = {
    "allow_fallbacks": _allow_fallbacks,
    "zdr": True,
}
if _provider_order_raw:
    DEFAULT_PROVIDER["order"] = _parse_provider_order(_provider_order_raw)

# Per-model overrides. A model not listed here uses DEFAULT_PROVIDER.
MODEL_PROVIDERS: dict[str, dict[str, Any]] = {}


# Field allow-lists
CHAT_ALLOWED_FIELDS = {
    "model",
    "models",
    "messages",
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "max_tokens",
    "stream",
    "tools",
    "tool_choice",
    "frequency_penalty",
    "presence_penalty",
    "stop",
    "n",
    "response_format",
    "seed",
    "reasoning",
}

EMBEDDING_ALLOWED_FIELDS = {
    "model",
    "models",
    "input",
    "encoding_format",
    "dimensions",
}


# Policy helpers
class PolicyError(ValueError):
    pass


def validate_model(body: dict[str, Any], restricted: frozenset[str]) -> None:
    requested: list[str] = []
    model = body.get("model")
    if isinstance(model, str):
        requested.append(model)
    models = body.get("models")
    if isinstance(models, list):
        requested.extend(str(m) for m in models if isinstance(m, str))

    if not requested:
        raise PolicyError("Missing required model selection.")

    blocked = [model for model in requested if model in restricted]
    if blocked:
        raise PolicyError(f"Restricted model(s): {', '.join(blocked)}")


def enforce_provider(body: dict[str, Any], model: str) -> dict[str, Any]:
    """Inject provider preferences into the request body.

    Uses a per-model override when available, otherwise DEFAULT_PROVIDER.
    Per-request ZDR is always re-applied, regardless of override.
    """
    result = deepcopy(body)
    prefs = dict(MODEL_PROVIDERS.get(model, DEFAULT_PROVIDER))
    prefs["zdr"] = True
    result["provider"] = prefs
    return result


def sanitize_body(body: dict[str, Any], endpoint: str) -> dict[str, Any]:
    allowed = CHAT_ALLOWED_FIELDS if endpoint == "chat" else EMBEDDING_ALLOWED_FIELDS
    return {k: v for k, v in body.items() if k in allowed}
