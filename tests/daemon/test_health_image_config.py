"""The public /api/health endpoint carries the client-side image re-encode config.

The composer (unauthenticated at page load) reads image_max_edge / image_quality
from here to downscale + re-encode photos before upload; health already exposes
version + agents, so it's reused rather than adding a new route.
"""

from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore


def _server(tmp_path, **http_kwargs):
    return HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374, **http_kwargs),
        adapters={},
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
        agent_configs={},
    )


def test_health_exposes_default_image_config(tmp_path):
    client = TestClient(_server(tmp_path).app)
    body = client.get("/api/health").json()
    assert body["images"] == {"max_edge": 1568, "quality": 0.85}


def test_health_reflects_configured_image_values(tmp_path):
    client = TestClient(_server(tmp_path, image_max_edge=1024, image_quality=0.7).app)
    body = client.get("/api/health").json()
    assert body["images"] == {"max_edge": 1024, "quality": 0.7}


def test_health_image_config_is_public(tmp_path):
    # No Authorization header: the composer needs it before the auth gate clears.
    resp = TestClient(_server(tmp_path).app).get("/api/health")
    assert resp.status_code == 200
