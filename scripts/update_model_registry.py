#!/usr/bin/env python3
"""Fetch the models.dev catalog and regenerate the model registry in provider files.

Source: https://models.dev - an open-source model catalog (140+ providers) with
structured capability metadata: reasoning options, modalities, context/output
limits, and per-million costs (including cache pricing, unused here for now).
Endpoint: https://models.dev/api.json - a dict keyed by provider id, each
carrying a `models` dict keyed by model id.

Usage:
    uv run python scripts/update_model_registry.py

The refresh is manual: run it when new models ship, review the diff of the
generated blocks, then run `uv run pytest` before committing. Only the code
between the BEGIN/END markers is rewritten; manual ModelInfo entries outside
the markers take priority at runtime, and the script warns when a generated
key duplicates one.
"""

import json
import re
import sys
from pathlib import Path

import httpx

from tsugite.providers.anthropic import _EFFORT_TO_BUDGET

MODELS_DEV_URL = "https://models.dev/api.json"

PROJECT_ROOT = Path(__file__).parent.parent
PROVIDER_FILES = {
    "openai": PROJECT_ROOT / "tsugite" / "providers" / "openai_compat.py",
    "anthropic": PROJECT_ROOT / "tsugite" / "providers" / "anthropic.py",
}

# models.dev provider id → our provider name
PROVIDER_MAP = {
    "openai": "openai",
    "anthropic": "anthropic",
}

# Model-id prefixes to skip (embeddings, media generation, moderation, etc.)
# that the output-modality check alone doesn't catch (e.g. embeddings report
# text output).
SKIP_PREFIXES = (
    "text-",
    "tts-",
    "whisper-",
    "dall-e",
    "gpt-image",
    "chatgpt-image",
    "omni-moderation",
    "computer-use-",
    "gpt-realtime",
    "gpt-audio",
    "sora-",
    "codex-",
)

# Tsugite's Anthropic provider translates effort strings into extended-thinking
# budget_tokens, so any model with a budget_tokens reasoning option supports
# that full vocabulary - even when models.dev lists a narrower native effort
# set (or none, e.g. the budget-only Haiku 4.5 / Claude 3.7 Sonnet). Derived
# from the runtime translation table so codegen can't drift from it.
BUDGET_TOKENS_EFFORT_LEVELS = list(_EFFORT_TO_BUDGET)


def fetch_models_dev_data() -> dict:
    print(f"Fetching {MODELS_DEV_URL}...")
    resp = httpx.get(MODELS_DEV_URL, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    data = resp.json()
    print(f"  Got {len(data)} providers")
    return data


def should_skip(key: str, entry: dict) -> bool:
    modalities = entry.get("modalities") or {}
    if "text" not in (modalities.get("output") or []):
        return True
    return any(key.startswith(p) for p in SKIP_PREFIXES)


def _effort_levels(entry: dict, provider: str) -> list[str] | None:
    """Derive supported_effort_levels from structured reasoning_options."""
    options = {o.get("type"): o for o in (entry.get("reasoning_options") or []) if isinstance(o, dict)}
    if provider == "anthropic" and "budget_tokens" in options:
        return list(BUDGET_TOKENS_EFFORT_LEVELS)
    effort = options.get("effort")
    if effort and effort.get("values"):
        return list(effort["values"])
    return None


def entry_to_model_info(key: str, entry: dict, provider: str) -> str:
    """Convert a models.dev entry to a ModelInfo(...) constructor string."""
    limit = entry.get("limit") or {}
    cost = entry.get("cost") or {}
    input_modalities = (entry.get("modalities") or {}).get("input") or []

    parts = []
    if limit.get("context"):
        parts.append(f"max_input_tokens={limit['context']:_}")
    if limit.get("output"):
        parts.append(f"max_output_tokens={limit['output']:_}")
    if cost.get("input"):
        parts.append(f"input_cost_per_million={round(float(cost['input']), 4)}")
    if cost.get("output"):
        parts.append(f"output_cost_per_million={round(float(cost['output']), 4)}")
    if "image" in input_modalities:
        parts.append("supports_vision=True")
    if "audio" in input_modalities:
        parts.append("supports_audio=True")
    if entry.get("reasoning"):
        parts.append("supports_reasoning=True")
    effort_levels = _effort_levels(entry, provider)
    if effort_levels:
        parts.append(f"supported_effort_levels={json.dumps(effort_levels)}")

    return f"ModelInfo({', '.join(parts)})"


def generate_dict_block(provider: str, models: dict[str, str]) -> str:
    """Generate the Python dict literal for a provider's models."""
    lines = []
    for key in sorted(models.keys()):
        lines.append(f'    "{provider}/{key}": {models[key]},')
    return "\n".join(lines)


def replace_generated_block(file_path: Path, provider: str, new_block: str) -> set[str]:
    """Replace the generated models block in a provider file. Returns set of manual model keys found."""
    content = file_path.read_text()

    begin_marker = f"# --- BEGIN GENERATED MODELS ({provider}) --- #"
    end_marker = f"# --- END GENERATED MODELS ({provider}) --- #"

    begin_idx = content.find(begin_marker)
    end_idx = content.find(end_marker)
    if begin_idx == -1 or end_idx == -1:
        print(f"  ERROR: Markers not found in {file_path}")
        sys.exit(1)

    # Find existing manual keys (outside the generated block) for duplicate detection
    before_block = content[:begin_idx]
    after_block = content[end_idx + len(end_marker) :]
    manual_keys = set()
    for match in re.finditer(r'"' + re.escape(provider) + r'/([^"]+)"', before_block + after_block):
        manual_keys.add(match.group(1))

    var_name = f"_{provider.upper()}_MODELS"
    replacement = (
        f"{begin_marker}\n"
        f"# Auto-generated by scripts/update_model_registry.py — do not edit manually\n"
        f"# fmt: off\n"
        f"{var_name}: dict[str, ModelInfo] = {{\n"
        f"{new_block}\n"
        f"}}\n"
        f"# fmt: on\n"
        f"{end_marker}"
    )

    new_content = content[:begin_idx] + replacement + content[end_idx + len(end_marker) :]
    file_path.write_text(new_content)
    return manual_keys


def main():
    data = fetch_models_dev_data()

    for dev_provider, provider in PROVIDER_MAP.items():
        entries = (data.get(dev_provider) or {}).get("models") or {}
        if not entries:
            print(f"  WARNING: no models found for provider '{dev_provider}'")
            continue
        models = {
            key: entry_to_model_info(key, entry, provider)
            for key, entry in entries.items()
            if not should_skip(key, entry)
        }

        print(f"\n{provider}: {len(models)} chat models found")

        file_path = PROVIDER_FILES[provider]
        manual_keys = replace_generated_block(file_path, provider, generate_dict_block(provider, models))

        # Check for duplicates
        dupes = manual_keys & set(models.keys())
        if dupes:
            print(f"  WARNING: {len(dupes)} duplicates with manual entries (manual takes priority at runtime):")
            for d in sorted(dupes):
                print(f"    - {provider}/{d}")

        print(f"  Wrote to {file_path.relative_to(PROJECT_ROOT)}")

    print("\nDone. Run `uv run pytest` to verify.")


if __name__ == "__main__":
    main()
