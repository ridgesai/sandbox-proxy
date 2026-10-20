# sandbox-proxy

## Logs

The proxy writes three files to `PROXY_DATA_DIR` (default `/proxy-data`).
They ship with the Harbor job artifact. Headers, including `Authorization`,
are never logged.

| File | Contents |
|---|---|
| `proxy_usage.json` | Cumulative cost snapshot for the run, rewritten after each successful call and at shutdown. |
| `proxy_inferences.jsonl` | One stats line per request. |
| `proxy_inference_payloads.jsonl` | One content line per request, joined to the stats line by `seq`. |

`seq` is `<boot id>-<n>`. Each proxy process has its own boot id, so values
never collide across processes or restarts.

### Stats line fields worth knowing

- `status_code` is what upstream returned; `returned_status_code` is present
  only when the agent got something else, for example a `429` once a response
  pushed the run over budget, or a `502` for an unusable upstream body.
- `error` is set for every failure, including errors OpenRouter reports
  inside a `200` body (top-level `error`, or `choices[].error` next to
  partial output). Those calls usually carry no usage, so their cost is 0.
- `provider` is the model vendor taken from the slug (`deepseek` for
  `deepseek/...`). `serving_provider` is who actually served the call, when
  upstream says so.
- `cached_tokens`, `reasoning_tokens` and `finish_reason` are `null` when the
  provider does not report them.

### Payload line

- `request` is the body sent upstream, after policy strips fields and injects
  provider preferences, `usage` and the seed. If the request was rejected
  before that rewrite, it is the agent's payload.
- `received_request` is the agent's payload as received, present whenever
  `request` is the rewritten body. Fields policy dropped (e.g. `logprobs`)
  survive only here.
- `response` is the full upstream body (`null` if the proxy rejected the
  request). Embedding vectors are replaced by their size, and a non-JSON body
  is kept as `{"raw_text": ...}`.
- `messages` (per message) and `tools` (as one blob) are stored as content
  hashes. A line defines a hash in `blobs` only the first time this process
  writes it, and lines are written in completion order, not `seq` order.
  **Readers must collect every `blobs` map in the file before resolving
  hashes.**
- Content logging stops once the file reaches `MAX_PAYLOAD_LOG_BYTES`
  (default 64 MiB). Later lines keep `seq`, status and error, and carry
  `truncated: true` with no request or response.
