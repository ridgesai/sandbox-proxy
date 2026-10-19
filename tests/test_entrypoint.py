from pathlib import Path


def test_docker_runtime_routes_http_and_https_to_one_metering_app() -> None:
    script = (Path(__file__).resolve().parents[1] / "entrypoint.sh").read_text()
    _, docker_branch = script.split("\nelse\n", maxsplit=1)

    assert docker_branch.count("exec uvicorn main:app") == 1
    assert docker_branch.count("uvicorn main:app") == 1
    assert "--port 443" in docker_branch
    assert "socat TCP-LISTEN:80,fork,reuseaddr" in docker_branch
    assert "OPENSSL:127.0.0.1:443,verify=0" in docker_branch


def test_docker_image_installs_http_relay() -> None:
    dockerfile = (Path(__file__).resolve().parents[1] / "Dockerfile").read_text()

    assert "    socat \\\n" in dockerfile


def test_kubernetes_runtime_topology_is_unchanged() -> None:
    script = (Path(__file__).resolve().parents[1] / "entrypoint.sh").read_text()
    kubernetes_branch, _ = script.split("\nelse\n", maxsplit=1)

    assert "--port 8080" in kubernetes_branch
    assert "--port 8443" in kubernetes_branch
    assert "exec python sni_router.py" in kubernetes_branch
