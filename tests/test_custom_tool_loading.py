"""Custom shell-tool loading must fail in the right direction.

`load_custom_shell_tools` runs inside `_ensure_tools_loaded`, before the plugin
loaders and before `_tools_loaded` latches. Anything it swallows or raises
affects every tool in the process, not just the custom ones.
"""

import sys
import types

import pytest


def test_a_malformed_config_is_reported_not_raised(monkeypatch, capsys, tmp_path):
    """Bad user YAML must not take down tool loading."""
    from tsugite.tools import load_custom_shell_tools

    config = tmp_path / "custom_tools.yaml"
    config.write_text("not: [valid")

    monkeypatch.setattr("tsugite.shell_tool_config.get_custom_tools_config_path", lambda: config)
    monkeypatch.setattr(
        "tsugite.shell_tool_config.load_custom_tools_config",
        lambda: (_ for _ in ()).throw(ValueError("bad yaml")),
    )

    load_custom_shell_tools()

    err = capsys.readouterr().err
    assert "Failed to load custom tools" in err
    assert str(config) in err, "the handler exists to name the config file"
    assert "tsugite tools validate" in err


def test_a_broken_install_is_not_disguised_as_bad_user_config(monkeypatch):
    """A first-party import failure must propagate.

    It used to be caught and reported as "check your config", which is actively
    misleading, and the handler then raised UnboundLocalError because the name it
    printed was imported inside the try it was recovering from.
    """
    from tsugite.tools import load_custom_shell_tools

    broken = types.ModuleType("tsugite.shell_tool_config")
    monkeypatch.setitem(sys.modules, "tsugite.shell_tool_config", broken)

    with pytest.raises(ImportError):
        load_custom_shell_tools()


def test_an_agents_custom_tool_is_bound_to_its_own_command(tmp_path):
    """`custom_tools` are registered into the process-global tool registry keyed
    by name, so two agents in one daemon that both define e.g. `deploy` share one
    slot. Re-resolving the run's tool by name therefore hands agent A the shell
    command agent B registered - arbitrary command substitution across agents.
    The run must bind its OWN definition.
    """
    import asyncio

    from tsugite.shell_tool_config import parse_tool_definition_from_dict
    from tsugite.tools.shell_tools import bind_custom_tools, register_shell_tools

    marker = tmp_path / "who_ran.txt"

    def define(who):
        return parse_tool_definition_from_dict(
            {
                "name": "deploy",
                "description": f"Deploy {who}'s service",
                "command": f"echo {who} > {marker}",
                "parameters": {},
            }
        )

    alice, bob = define("alice"), define("bob")

    # Agent alice starts a turn and binds its own `deploy`, the way runner.py does.
    (alice_tool,) = bind_custom_tools([alice])

    # Agent bob runs at any point afterwards and takes over the registry slot.
    register_shell_tools([bob])

    asyncio.run(alice_tool.execute())
    assert marker.read_text().strip() == "alice", "alice's deploy tool ran another agent's command"
