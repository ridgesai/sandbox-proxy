# sandbox-proxy

OpenRouter enforcement proxy sidecar for Ridges evaluation sandboxes.

Runs alongside agent code during evaluations to:
- Enforce cost budgets (`MAX_COST_USD`)
- Validate and sanitize LLM requests (model allow/block lists, field sanitization)
- Apply provider preferences (ZDR, routing)
- Log per-inference usage to `/proxy-data/proxy_inferences.jsonl`
- Block all non-OpenRouter HTTPS egress during the agent phase; allow full egress during verification

## Execution modes

The proxy detects its execution context at runtime via `id -u`:

### Docker Compose (root, ports 80/443)

Harbor's compose scaffold sets `openrouter.ai` as a network alias for this container. All HTTPS traffic from the agent container destined for `openrouter.ai` arrives here directly on port 443.

```
agent container → openrouter.ai:443 (network alias) → proxy:443
```

### Kubernetes (UID 1337, ports 8080/8443/15443)

An `iptables-init` container redirects all outbound TCP 443 → 15443 (except UID 1337, which is exempt). The SNI router on 15443 inspects the TLS ClientHello and routes:

- `openrouter.ai` → local MITM proxy on 127.0.0.1:8443
- Any other host (agent phase) → **blocked**
- Any other host (verification phase, after `/tmp/egress-unlocked` is touched) → transparent TCP tunnel

```
agent container → :443 → iptables REDIRECT → proxy:15443 (SNI router)
                                                    ├── openrouter.ai → :8443 (MITM)
                                                    └── other → tunnel or block
```

## Environment variables

| Variable | Description | Default |
|---|---|---|
| `EVALUATION_RUN_ID` | Identifier for the current eval run | `unknown-eval-run` |
| `MAX_COST_USD` | Hard budget cap in USD | `9999.0` |
| `UPSTREAM_BASE_URL` | OpenRouter base URL | `https://openrouter.ai` |
| `OPENROUTER_MANAGEMENT_KEY` | Management API key for workspace policy checks | — |
| `OPENROUTER_WORKSPACE_ID` | Workspace ID for privacy policy enforcement | — |
| `OPENROUTER_EXPECTED_API_KEY_SHA256` | SHA-256 of the runtime API key to verify | — |



