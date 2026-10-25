"""Reading a verifier's answer: parse its JSON verdict and classify infrastructure failures.

Pure parsing - no orchestrator state, no I/O.
"""

import json
import re
from typing import Iterator, Optional

_VERIFIER_EXCERPT_LIMIT = 200

_INFRA_FAILURE_RE = re.compile(
    r"usage.?limit|rate.?limit|quota|\b429\b|too many requests|overloaded|"
    r"insufficient credit|resource.?exhausted|capacity",
    re.IGNORECASE,
)


def _is_infra_failure(text: str) -> bool:
    """Usage/rate-limit/quota style failures are infrastructure, not quality:
    they must not read as an acceptance-criteria problem and are the natural
    trigger for escalating to a different model."""
    return bool(text and _INFRA_FAILURE_RE.search(text))


def _sanitize_output_excerpt(raw: str) -> str:
    """Bounded single-line excerpt of verifier output for error diagnostics.

    Masks registered secrets BEFORE truncating - truncating first could split a
    secret so the mask no longer recognises it, leaking the remainder.
    """
    from tsugite.secrets.registry import get_registry

    text = get_registry().mask(" ".join((raw or "").split()))
    if len(text) > _VERIFIER_EXCERPT_LIMIT:
        text = text[:_VERIFIER_EXCERPT_LIMIT] + "…"
    return text or "(empty)"


def _parse_verifier_output(raw: str) -> Optional[dict]:
    """Extract the verifier's JSON verdict from its output text.

    Verifier output is model text, not guaranteed clean JSON: real sessions have
    produced junk-JSON preambles (`{"cmd": "dummy"}{}...` tool-call flailing),
    markdown-fenced verdicts, prose around the object, and the verdict emitted
    twice back-to-back. Scan every top-level JSON object in the text and prefer
    the LAST one that looks like a verdict (has `ac_results` or `overall_pass`);
    fall back to the first object so bare single-object output still parses.

    Returns None when no JSON object is found at all (empty input, prose-only,
    or non-object JSON like `42` / `[]`) - `_handle_verifier_complete` must be
    able to do `parsed.get(...)` on the return value.
    """
    first_obj: Optional[dict] = None
    last_verdict: Optional[dict] = None
    for value in _json_objects(raw):
        if first_obj is None:
            first_obj = value
        if "ac_results" in value or "overall_pass" in value:
            last_verdict = value
    return last_verdict if last_verdict is not None else first_obj


def _json_objects(raw: str) -> Iterator[dict]:
    """Every top-level JSON object embedded in `raw`, in order."""
    decoder = json.JSONDecoder()
    idx = 0
    while raw and (start := raw.find("{", idx)) != -1:
        try:
            value, end = decoder.raw_decode(raw, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(value, dict):
            yield value
        idx = end
