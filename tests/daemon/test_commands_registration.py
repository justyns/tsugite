"""Guard against unsupported CommandParam types breaking adapter command registration.

The Discord adapter (`tsugite/daemon/adapters/discord.py`) feeds `CommandParam.type`
directly into a `discord.app_commands.Command` callback annotation. Discord
slash-commands only accept scalar option types (str/int/bool/float/User/etc) —
container types like `list` or `dict` make registration crash at bot startup.

Caught a regression: an earlier `/job` definition used `list` for the
`acceptance_criteria` param.
"""

import pytest

from tsugite.daemon.commands import get_commands

_DISCORD_SAFE_PARAM_TYPES = {str, int, bool, float}


@pytest.mark.parametrize(
    "cmd_name,param_name,param_type",
    [(cmd.name, p.name, p.type) for cmd in get_commands().values() for p in cmd.params],
)
def test_all_command_param_types_are_discord_compatible(cmd_name, param_name, param_type):
    assert param_type in _DISCORD_SAFE_PARAM_TYPES, (
        f"Command '/{cmd_name}' param '{param_name}' has type {param_type!r}, which "
        f"the Discord adapter cannot register. Use str (and parse internally) "
        f"for container values. Allowed: {_DISCORD_SAFE_PARAM_TYPES}."
    )
