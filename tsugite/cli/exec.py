"""CLI exec command - run Python in tsugite's tool namespace.

Lets an external caller (e.g. Claude Code) reuse a tsugite skill's Python verbatim:
tsugite tools are available as functions, secrets stay allowlisted + masked, and the
snippet can be sandboxed. This is the execution half of the Claude Code <-> tsugite
bridge; the agent loop's `SubprocessExecutor` is reused so behavior matches `tsu run`.
"""

import asyncio
from pathlib import Path
from typing import List, Optional

import typer

from tsugite.options import ExecutionOptions

# Sensible default tool surface when neither --tools nor --agent is given.
DEFAULT_TOOL_SPECS = ["@fs", "@http", "@secrets"]


async def _execute_bounded(executor, code: str, timeout: int):
    return await asyncio.wait_for(executor.execute(code), timeout)


def exec_cmd(
    script: Optional[str] = typer.Argument(None, help="Python file to run, or '-' for stdin (default: stdin)"),
    agent: Optional[str] = typer.Option(
        None, "--agent", "-a", help="Inherit an agent's tools + allowed_secrets (e.g. +assistant or path.md)"
    ),
    tools: Optional[List[str]] = typer.Option(
        None,
        "--tools",
        "-t",
        help=f"Tool specs to expose; repeat the flag or comma/space-separate (e.g. '@fs,@http'). Default: {' '.join(DEFAULT_TOOL_SPECS)}",
    ),
    allow_secret: Optional[List[str]] = typer.Option(
        None, "--allow-secret", help="Secret name(s) the snippet may read (repeatable; none given = all allowed)"
    ),
    timeout: int = typer.Option(30, "--timeout", help="Wall-clock timeout in seconds"),
    workspace: Optional[str] = typer.Option(
        None, "--workspace", "-w", help="Working directory for the snippet and its tools"
    ),
    sandbox: bool = typer.Option(False, "--sandbox", help="Run the snippet in a bubblewrap sandbox"),
    no_sandbox: bool = typer.Option(False, "--no-sandbox", help="Disable sandbox (overrides config)"),
    allow_domain: Optional[List[str]] = typer.Option(
        None, "--allow-domain", help="Domain(s) allowed in sandbox (implies --sandbox)"
    ),
    no_network: bool = typer.Option(False, "--no-network", help="Sandbox with no network at all (implies --sandbox)"),
    no_secrets: bool = typer.Option(False, "--no-secrets", help="Skip secrets backend initialization"),
):
    """Run a Python snippet in tsugite's tool namespace.

    tsugite tools (read_file, http_request, get_secret, ...) are available as
    functions, exactly as in an agent's code block - so you can reuse a tsugite
    skill's Python verbatim from outside tsugite.

    Notes:
      - open() is blocked; use read_file/write_file instead.
      - Secret values are masked in output. Access is gated by --agent's
        allowed_secrets or --allow-secret (no restriction given = all allowed).
      - --no-network / --allow-domain imply --sandbox.

    Examples:
        echo 'read_file("README.md")' | tsu exec -
        tsu exec snippet.py --tools @fs,@http
        tsu exec snippet.py --agent +assistant
        tsu exec snippet.py --allow-secret gh-token --no-network
    """
    from tsugite.agent_runner.helpers import (
        build_sandbox_policy,
        clear_current_agent,
        clear_sandbox_context,
        set_allowed_secrets,
        set_current_agent,
        set_sandbox_context,
    )
    from tsugite.core.subprocess_executor import SubprocessExecutor
    from tsugite.core.tools import create_tool_from_tsugite
    from tsugite.secrets import init_cli as init_secrets
    from tsugite.secrets.registry import get_registry
    from tsugite.tools import expand_tool_specs
    from tsugite.utils import read_stdin

    init_secrets(no_secrets)

    code = read_stdin() if script in (None, "-") else Path(script).read_text()
    if not code.strip():
        typer.echo("No code provided.", err=True)
        raise typer.Exit(1)

    # Optional agent context: inherit its tools + allowed_secrets + sandbox frontmatter.
    agent_config = None
    if agent:
        from tsugite.cli.helpers import load_and_validate_agent
        from tsugite.console import get_stderr_console

        loaded, _, _ = load_and_validate_agent(agent, get_stderr_console())
        agent_config = loaded.config

    agent_name = agent_config.name if agent_config else "exec"

    if tools:
        # Accept repeated flags plus comma/space-separated specs in one value, e.g.
        # `--tools @fs,@http` or `--tools "@fs @http"` (a bare `--tools @fs @http`
        # can't work: the second token is parsed as a positional arg).
        tool_specs = [spec for value in tools for spec in value.replace(",", " ").split()]
    elif agent_config and agent_config.tools:
        tool_specs = agent_config.tools
    else:
        tool_specs = DEFAULT_TOOL_SPECS

    if allow_secret is not None:
        allowed_secrets = list(allow_secret)
    elif agent_config:
        allowed_secrets = list(agent_config.allowed_secrets)
    else:
        allowed_secrets = []

    try:
        tool_objs = [create_tool_from_tsugite(name) for name in expand_tool_specs(tool_specs)]
    except (ValueError, KeyError) as e:
        typer.echo(f"Tool resolution failed: {e}", err=True)
        raise typer.Exit(1)

    # --no-network / --allow-domain are isolation knobs; enabling them without
    # --sandbox would silently do nothing, so they imply the sandbox here.
    effective_sandbox = sandbox or no_network or bool(allow_domain)
    exec_opts = ExecutionOptions.from_cli(
        sandbox=effective_sandbox,
        no_sandbox=no_sandbox,
        allow_domain=allow_domain,
        no_network=no_network,
    )

    workspace_dir = Path(workspace).expanduser().resolve() if workspace else None

    try:
        sandbox_config, sandbox_ctx = build_sandbox_policy(
            exec_opts, workspace_dir=workspace_dir, agent_config=agent_config
        )
    except RuntimeError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    set_current_agent(agent_name)
    set_allowed_secrets(allowed_secrets)
    set_sandbox_context(sandbox_ctx)

    executor = SubprocessExecutor(workspace_dir=workspace_dir, sandbox_config=sandbox_config)
    executor.set_tools(tool_objs)

    mask = get_registry().mask
    try:
        result = asyncio.run(_execute_bounded(executor, code, timeout))
    except TimeoutError:
        typer.echo(f"Execution timed out after {timeout}s", err=True)
        raise typer.Exit(1)
    finally:
        executor.cleanup()
        # Don't leak the thread-local agent/sandbox context past this one-shot run
        # (the agent runner clears these in its finally for the same reason).
        clear_sandbox_context()
        clear_current_agent()

    if result.error:
        typer.echo(mask(result.error), err=True)
        tb = (result.stderr or "").strip()
        if tb and tb != (result.error or "").strip():
            typer.echo(mask(tb), err=True)
        raise typer.Exit(1)

    out = result.output or ""
    if out:
        typer.echo(mask(out), nl=False)
    elif result.return_value is not None:
        typer.echo(mask(str(result.return_value)))
