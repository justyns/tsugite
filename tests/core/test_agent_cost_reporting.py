"""Cost semantics: a provider-reported $0.00 (subscription models like
codex_cli) must be recorded as 0.0, while "no cost data at all" (e.g. an
interrupted claude_code turn) stays None. The old `total_cost > 0 else None`
coercion erased that distinction and left NULL cost_usd rows in usage.db for
turns with real token counts."""

from tsugite.core.agent import TsugiteAgent
from tsugite.providers.base import Usage


def _bare_agent() -> TsugiteAgent:
    a = object.__new__(TsugiteAgent)
    a.total_tokens = 0
    a.total_cost = 0.0
    a.cost_reported = False
    a.last_input_tokens = 0
    a.cache_creation_tokens = 0
    a.cache_read_tokens = 0
    return a


def _usage(total=100):
    return Usage(prompt_tokens=60, completion_tokens=40, total_tokens=total)


def test_provider_reported_zero_cost_is_zero_not_none():
    a = _bare_agent()
    a._accumulate_usage(_usage(), 0.0)
    assert a.reported_cost == 0.0, "a real $0 (subscription provider) must not degrade to None"


def test_no_cost_data_stays_none():
    a = _bare_agent()
    a._accumulate_usage(_usage(), None)
    assert a.reported_cost is None, "tokens without any provider cost signal are unknown, not $0"


def test_never_ran_is_none():
    a = _bare_agent()
    assert a.reported_cost is None


def test_real_cost_passes_through():
    a = _bare_agent()
    a._accumulate_usage(_usage(), 1.25)
    a._accumulate_usage(_usage(), 0.75)
    assert a.reported_cost == 2.0


def test_mixed_none_and_reported_still_reports():
    a = _bare_agent()
    a._accumulate_usage(_usage(), None)
    a._accumulate_usage(_usage(), 0.5)
    assert a.reported_cost == 0.5
