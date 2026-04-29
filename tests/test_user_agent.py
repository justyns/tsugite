"""set_user_agent_header forces the framework UA on outbound requests.

Background: agent code that sets headers["User-Agent"] = "..." has been
observed leaking PII (e.g. user emails copied from system-context into a
contact-style UA). The framework now overwrites and warns instead of
deferring to the caller.
"""

import logging
from unittest.mock import patch

from tsugite.user_agent import set_user_agent_header


FRAMEWORK_UA = "Tsugite/test (+https://github.com/justyns/tsugite)"


class TestForcesFrameworkUA:
    def test_no_caller_ua_sets_framework(self):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {}
            set_user_agent_header(headers)
        assert headers == {"User-Agent": FRAMEWORK_UA}

    def test_caller_ua_overwritten_and_warned(self, caplog):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"User-Agent": "agent-leaks-pii (user@example.com)"}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers["User-Agent"] == FRAMEWORK_UA
        assert "Dropped caller-supplied User-Agent" in caplog.text
        assert "user@example.com" in caplog.text

    def test_caller_ua_lowercase_overwritten_and_warned(self, caplog):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"user-agent": "leaky-bot/1.0 (REDACT)"}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers == {"User-Agent": FRAMEWORK_UA}
        assert "leaky-bot/1.0" in caplog.text

    def test_caller_ua_uppercase_overwritten_and_warned(self, caplog):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"USER-AGENT": "shouty/1.0"}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers == {"User-Agent": FRAMEWORK_UA}
        assert "shouty/1.0" in caplog.text

    def test_caller_ua_matches_framework_no_warning(self, caplog):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"User-Agent": FRAMEWORK_UA}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers == {"User-Agent": FRAMEWORK_UA}
        assert caplog.text == ""

    def test_no_caller_ua_no_warning(self, caplog):
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"Accept": "application/json"}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers["User-Agent"] == FRAMEWORK_UA
        assert headers["Accept"] == "application/json"
        assert caplog.text == ""

    def test_disabled_ua_skips_caller_too(self):
        """When UA management is disabled (config user_agent=""), the function
        is a no-op. Caller-supplied UA is left alone — disable means hands-off.
        """
        with patch("tsugite.user_agent.get_user_agent", return_value=None):
            headers = {"User-Agent": "caller/1.0"}
            set_user_agent_header(headers)
        assert headers == {"User-Agent": "caller/1.0"}

    def test_disabled_ua_no_op_when_empty(self):
        with patch("tsugite.user_agent.get_user_agent", return_value=None):
            headers = {}
            set_user_agent_header(headers)
        assert headers == {}

    def test_logged_caller_ua_truncated(self, caplog):
        """Long caller UAs get truncated in the log message to keep logs sane."""
        long_value = "x" * 500
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"User-Agent": long_value}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert headers["User-Agent"] == FRAMEWORK_UA
        assert long_value not in caplog.text
        assert "x" * 100 in caplog.text

    def test_multiple_case_variants_all_dropped(self, caplog):
        """If somehow a dict has both 'User-Agent' and 'user-agent', both go."""
        with patch("tsugite.user_agent.get_user_agent", return_value=FRAMEWORK_UA):
            headers = {"User-Agent": "first", "user-agent": "second"}
            with caplog.at_level(logging.WARNING, logger="tsugite.user_agent"):
                set_user_agent_header(headers)
        assert list(headers.keys()) == ["User-Agent"]
        assert headers["User-Agent"] == FRAMEWORK_UA
