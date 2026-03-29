"""OpenRouter provider — OpenAI-compatible with rich model metadata from their API."""

from __future__ import annotations

import logging
from typing import Any

from .base import ModelInfo
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class OpenRouterProvider(OpenAICompatProvider):
    async def list_models(self) -> list[str]:
        from .model_registry import register_model

        client = self._get_client()
        headers = self._build_headers()

        resp = await client.get(f"{self.api_base}/models", headers=headers)
        resp.raise_for_status()
        models = resp.json().get("data", [])

        chat_models = []
        for m in models:
            model_id = m.get("id", "")
            arch = m.get("architecture", {})
            input_mods = arch.get("input_modalities", [])
            output_mods = arch.get("output_modalities", [])

            if "text" not in output_mods:
                continue

            pricing = m.get("pricing", {})
            top = m.get("top_provider", {})

            prompt_cost = float(pricing.get("prompt", 0))
            completion_cost = float(pricing.get("completion", 0))

            # Negative prices are OpenRouter's sentinel for "variable/unknown"
            if prompt_cost < 0:
                prompt_cost = 0
            if completion_cost < 0:
                completion_cost = 0

            info = ModelInfo(
                max_input_tokens=m.get("context_length") or 128_000,
                max_output_tokens=top.get("max_completion_tokens"),
                input_cost_per_million=round(prompt_cost * 1_000_000, 4) if prompt_cost else None,
                output_cost_per_million=round(completion_cost * 1_000_000, 4) if completion_cost else None,
                supports_vision="image" in input_mods,
                supports_audio="audio" in input_mods,
            )
            register_model("openrouter", model_id, info)
            chat_models.append(model_id)

        return chat_models


def create_provider(name: str = "openrouter", **kwargs: Any) -> OpenRouterProvider:
    import os

    return OpenRouterProvider(
        name=name,
        api_base=kwargs.get("api_base") or "https://openrouter.ai/api/v1",
        api_key=kwargs.get("api_key") or os.getenv("OPENROUTER_API_KEY"),
        extra_headers=kwargs.get("extra_headers", {}),
    )
