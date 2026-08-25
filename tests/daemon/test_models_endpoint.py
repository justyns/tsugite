"""/api/models serializes the registry's pricing + capability metadata.

The header model picker groups by provider and shows per-model context window,
input/output price, and vision/reasoning badges, so the endpoint must carry those
fields. Pricing is nullable: CLI / openai-compat models register with no cost.
"""

from starlette.testclient import TestClient
from tsugite_daemon.adapters.http import HTTPServer
from tsugite_daemon.config import HTTPConfig
from tsugite_daemon.webhook_store import WebhookStore

from tsugite.providers.model_registry import get_model_info


def _models_by_id(tmp_path):
    server = HTTPServer(
        config=HTTPConfig(enabled=True, host="127.0.0.1", port=8374),
        adapter=None,
        webhook_store=WebhookStore(tmp_path / "webhooks.json"),
    )
    body = TestClient(server.app).get("/api/models").json()
    return {m["id"]: m for m in body["models"]}


def test_models_endpoint_serves_pricing_and_limits(tmp_path):
    models = _models_by_id(tmp_path)
    m = models["anthropic:claude-opus-4-5"]
    # Compare against the registry rather than hardcoded prices so a price update
    # doesn't break the test - what's under test is that the attributes flow.
    info = get_model_info("anthropic", "claude-opus-4-5")
    assert info is not None
    assert m["input_cost_per_million"] == info.input_cost_per_million
    assert m["output_cost_per_million"] == info.output_cost_per_million
    assert m["max_output_tokens"] == info.max_output_tokens
    # A known priced model must serialize non-null pricing (regression guard).
    assert m["input_cost_per_million"] is not None
    # Existing fields stay.
    assert m["context_window"] == info.max_input_tokens
    assert m["supports_vision"] is True
    assert m["supports_reasoning"] is True


def test_models_endpoint_always_carries_pricing_keys(tmp_path):
    # Every model serializes the pricing/limit keys even when unpriced, so the
    # frontend reads them without a presence check.
    models = _models_by_id(tmp_path)
    assert models  # priming registered at least the anthropic table
    for m in models.values():
        assert "input_cost_per_million" in m
        assert "output_cost_per_million" in m
        assert "max_output_tokens" in m
