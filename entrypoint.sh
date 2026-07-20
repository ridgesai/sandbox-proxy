#!/usr/bin/env sh
set -eu

mkdir -p /certs/output
mkdir -p "${PROXY_DATA_DIR:-/proxy-data}"

# Build trust bundle for the main container.
cat /etc/ssl/certs/ca-certificates.crt /certs/ca.crt > /certs/output/ca-bundle.crt
cp /certs/ca.crt /certs/output/ridges-ca.crt

if [ "$(id -u)" -ne 0 ]; then
  # Kubernetes mode: high ports + SNI router.
  uvicorn main:app --host 0.0.0.0 --port 8080 &
  uvicorn main:app --host 127.0.0.1 --port 8443 \
    --ssl-keyfile /certs/server.key \
    --ssl-certfile /certs/server.crt &
  exec python sni_router.py
else
  # Docker Compose mode: pin real IP to bypass the openrouter.ai DNS alias.
  REAL_IP=$(dig +short openrouter.ai @8.8.8.8 A | grep -E '^[0-9]' | head -1 || true)
  if [ -n "$REAL_IP" ]; then
    echo "$REAL_IP openrouter.ai" >> /etc/hosts
  fi

  uvicorn main:app --host 0.0.0.0 --port 80 &
  exec uvicorn main:app --host 0.0.0.0 --port 443 \
    --ssl-keyfile /certs/server.key \
    --ssl-certfile /certs/server.crt
fi
