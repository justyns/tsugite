"""Slice 1: alias parsing for the ACP provider."""

from __future__ import annotations

import pytest


class TestAliases:
    def test_short_aliases_resolve(self):
        from tsugite_acp.models import resolve_model_alias

        assert resolve_model_alias("opus") == "claude-opus-4-8"
        assert resolve_model_alias("sonnet") == "claude-sonnet-4-6"
        assert resolve_model_alias("haiku") == "claude-haiku-4-5-20251001"

    def test_full_id_passthrough(self):
        from tsugite_acp.models import resolve_model_alias

        assert resolve_model_alias("claude-sonnet-4-6") == "claude-sonnet-4-6"
        assert resolve_model_alias("claude-opus-4-7") == "claude-opus-4-7"

    def test_unknown_string_passthrough(self):
        from tsugite_acp.models import resolve_model_alias

        assert resolve_model_alias("some-future-model") == "some-future-model"

    def test_empty_string_raises(self):
        from tsugite_acp.models import resolve_model_alias

        with pytest.raises(ValueError):
            resolve_model_alias("")


class TestModelRegistration:
    def test_register_models_populates_registry(self):
        from tsugite_acp.models import register_acp_models

        from tsugite.providers.model_registry import get_model_info

        register_acp_models()

        info = get_model_info("acp", "claude-sonnet-4-6")
        assert info is not None
        assert info.supports_vision is True
        assert info.max_input_tokens >= 200_000

    def test_aliases_cover_all_registered_models(self):
        from tsugite_acp.models import _ACP_MODELS, _ALIASES

        for full_id in _ALIASES.values():
            assert any(full_id in key for key in _ACP_MODELS), f"{full_id} missing from _ACP_MODELS"
