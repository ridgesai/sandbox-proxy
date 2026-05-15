FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    openssl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

RUN mkdir -p /certs /certs/output && \
    openssl genrsa -out /certs/ca.key 2048 && \
    openssl req -new -x509 -days 3650 -key /certs/ca.key \
      -out /certs/ca.crt -subj "/CN=Ridges Proxy CA" && \
    openssl genrsa -out /certs/server.key 2048 && \
    printf "subjectAltName=DNS:openrouter.ai\n" > /tmp/san.ext && \
    openssl req -new -key /certs/server.key \
      -out /certs/server.csr -subj "/CN=openrouter.ai" && \
    openssl x509 -req -days 3650 -in /certs/server.csr \
      -CA /certs/ca.crt -CAkey /certs/ca.key -CAcreateserial \
      -out /certs/server.crt -extfile /tmp/san.ext && \
    rm -f /certs/server.csr /certs/ca.srl /tmp/san.ext && \
    chmod -R 755 /certs

RUN adduser --uid 1337 --disabled-password --gecos "" ridges-proxy && \
    chown -R ridges-proxy:ridges-proxy /certs/output

COPY . /app
WORKDIR /app
RUN chmod +x /app/entrypoint.sh

EXPOSE 80 443 8080 8443 15443

CMD ["/app/entrypoint.sh"]
