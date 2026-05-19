#!/usr/bin/env sh
set -eu

mkdir -p /certs/output
mkdir -p "${PROXY_DATA_DIR:-/proxy-data}"

# Trust bundle used by Python/Requests callers in the main container.
cat /etc/ssl/certs/ca-certificates.crt /certs/ca.crt > /certs/output/ca-bundle.crt
cp /certs/ca.crt /certs/output/ridges-ca.crt

if [ "$(id -u)" -ne 0 ]; then
  # Kubernetes mode (UID 1337): high ports + SNI router.
  # iptables (set up by the iptables-init container) redirects :443 -> :15443.
  uvicorn main:app --host 0.0.0.0 --port 8080 &

  # HTTPS enforcement proxy on loopback only — the SNI router forwards
  # openrouter.ai traffic here; all other HTTPS is tunnelled transparently.
  uvicorn main:app --host 127.0.0.1 --port 8443 \
    --ssl-keyfile /certs/server.key \
    --ssl-certfile /certs/server.crt &

  # SNI-aware TCP router on port 15443 (iptables REDIRECT target).
  exec python sni_router.py
else
  # Docker Compose mode (root): standard low ports, no SNI router needed.
  # Docker's network alias resolves openrouter.ai to this container, so all
  # HTTPS traffic destined for openrouter.ai arrives here directly on :443.
  #
  # Pin the real OpenRouter IP so the proxy's own upstream calls (workspace
  # policy check) bypass the Docker DNS alias that would otherwise loop back
  # to this container. dig @8.8.8.8 bypasses Docker's embedded DNS
  # (127.0.0.11) entirely. /etc/hosts takes precedence over DNS for all
  # subsequent lookups, so this resolves before uvicorn starts serving.
  REAL_IP=$(dig +short openrouter.ai @8.8.8.8 A | grep -E '^[0-9]' | head -1 || true)
  if [ -n "$REAL_IP" ]; then
    echo "$REAL_IP openrouter.ai" >> /etc/hosts
  fi

  uvicorn main:app --host 0.0.0.0 --port 80 &
  exec uvicorn main:app --host 0.0.0.0 --port 443 \
    --ssl-keyfile /certs/server.key \
    --ssl-certfile /certs/server.crt
fi
