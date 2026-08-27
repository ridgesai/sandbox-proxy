FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    dnsutils \
    openssl \
    socat \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /certs /certs/output && \
    openssl genrsa -out /certs/ca.key 2048 && \
    printf '%s\n' \
      '[req]' \
      'distinguished_name = dn' \
      'x509_extensions = v3_ca' \
      'prompt = no' \
      '' \
      '[dn]' \
      'CN = Ridges Proxy CA' \
      '' \
      '[v3_ca]' \
      'basicConstraints = critical, CA:TRUE' \
      'keyUsage = critical, keyCertSign, cRLSign' \
      'subjectKeyIdentifier = hash' \
      > /tmp/ca.conf && \
    openssl req -new -x509 -days 3650 -key /certs/ca.key \
      -out /certs/ca.crt -config /tmp/ca.conf && \
    openssl genrsa -out /certs/server.key 2048 && \
    printf '%s\n' \
      '[req]' \
      'distinguished_name = dn' \
      'prompt = no' \
      '' \
      '[dn]' \
      'CN = openrouter.ai' \
      > /tmp/server.conf && \
    printf '%s\n' \
      'basicConstraints = critical, CA:FALSE' \
      'keyUsage = critical, digitalSignature, keyEncipherment' \
      'extendedKeyUsage = serverAuth' \
      'subjectAltName = DNS:openrouter.ai' \
      'authorityKeyIdentifier = keyid,issuer' \
      'subjectKeyIdentifier = hash' \
      > /tmp/server.ext && \
    openssl req -new -key /certs/server.key \
      -out /certs/server.csr -config /tmp/server.conf && \
    openssl x509 -req -days 3650 -in /certs/server.csr \
      -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial \
      -out /certs/server.crt -extfile /tmp/server.ext && \
    rm -f /certs/server.csr /certs/ca.srl /tmp/ca.conf /tmp/server.conf /tmp/server.ext && \
    chmod -R 755 /certs

RUN adduser --uid 1337 --disabled-password --gecos "" ridges-proxy && \
    chown -R ridges-proxy:ridges-proxy /certs/output

COPY . /app
WORKDIR /app
RUN chmod +x /app/entrypoint.sh

EXPOSE 80 443 8080 8443 15443

CMD ["/app/entrypoint.sh"]
