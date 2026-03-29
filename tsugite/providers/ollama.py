"""Ollama provider — OpenAI-compatible with rich model discovery via /api/show."""

from __future__ import annotations

import logging
from typing import Any

from .base import ModelInfo
from .openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class OllamaProvider(OpenAICompatProvider):
    async def list_models(self) -> list[str]:
        from .model_registry import register_model

        client = self._get_client()
        headers = self._build_headers()
        base = self.api_base.replace("/v1", "")

        resp = await client.get(f"{base}/api/tags", headers=headers)
        resp.raise_for_status()
        all_names = [m["name"] for m in resp.json().get("models", [])]

        chat_models = []
        for name in all_names:
            try:
                show_resp = await client.post(f"{base}/api/show", json={"name": name}, headers=headers)
                if show_resp.status_code != 200:
                    chat_models.append(name)
                    continue

                data = show_resp.json()
                caps = data.get("capabilities") or []

                if caps and "completion" not in caps:
                    continue

                model_info_dict = data.get("model_info", {})

                ctx_len = None
                for key, val in model_info_dict.items():
                    if key.endswith(".context_length") and isinstance(val, int):
                        ctx_len = val
                        break

                if not ctx_len:
                    for line in data.get("parameters", "").splitlines():
                        if "num_ctx" in line:
                            try:
                                ctx_len = int(line.split()[-1])
                            except (ValueError, IndexError):
                                pass

                info = ModelInfo(
                    max_input_tokens=ctx_len or 128_000,
                    supports_vision="vision" in caps,
                )
                register_model("ollama", name, info)
                chat_models.append(name)
            except Exception:
                chat_models.append(name)

        return chat_models


def create_provider(name: str = "ollama", **kwargs: Any) -> OllamaProvider:
    import os

    return OllamaProvider(
        name=name,
        api_base=kwargs.get("api_base") or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        api_key=kwargs.get("api_key") or "ollama",
        extra_headers=kwargs.get("extra_headers", {}),
    )
