"""Main daemon gateway coordinating all adapters."""

import asyncio
from pathlib import Path
from typing import List, Optional

from tsugite.daemon.adapters.base import BaseAdapter, resolve_agent_path
from tsugite.daemon.config import DaemonConfig, load_daemon_config
from tsugite.daemon.session import SessionManager


class Gateway:
    """Main daemon gateway routing messages between platform adapters and agents."""

    def __init__(self, config: DaemonConfig):
        """Initialize gateway.

        Args:
            config: Daemon configuration
        """
        self.config = config
        self.adapters: List[BaseAdapter] = []

    async def start(self):
        """Start all enabled adapters."""

        # Start Discord bots (one adapter per bot)
        if self.config.discord_bots:
            try:
                from tsugite.daemon.adapters.discord import DiscordAdapter
            except ImportError as e:
                raise ImportError(
                    "Discord support requires discord.py. Install with: pip install tsugite-cli[daemon]"
                ) from e

            for bot_config in self.config.discord_bots:
                agent_name = bot_config.agent
                if agent_name not in self.config.agents:
                    raise ValueError(f"Discord bot '{bot_config.name}' references unknown agent '{agent_name}'")

                agent_config = self.config.agents[agent_name]

                # Validate agent file exists at startup
                from tsugite.workspace import Workspace, WorkspaceNotFoundError

                try:
                    workspace = Workspace.load(agent_config.workspace_dir)
                except WorkspaceNotFoundError:
                    workspace = None

                agent_path = resolve_agent_path(agent_config.agent_file, agent_config.workspace_dir, workspace)
                if not agent_path:
                    raise ValueError(
                        f"Agent file '{agent_config.agent_file}' not found for bot '{bot_config.name}'. "
                        f"Searched in workspace '{agent_config.workspace_dir}' and standard paths."
                    )
                print(f"  ✓ Bot '{bot_config.name}' using agent: {agent_path}")

                session_manager = SessionManager(
                    agent_name, agent_config.workspace_dir, context_limit=agent_config.context_limit
                )

                discord_adapter = DiscordAdapter(
                    bot_config=bot_config,
                    agent_name=agent_name,
                    agent_config=agent_config,
                    session_manager=session_manager,
                )
                self.adapters.append(discord_adapter)

        if not self.adapters:
            raise ValueError("No adapters enabled in config")

        print(f"Starting {len(self.adapters)} adapter(s)...")
        try:
            await asyncio.gather(*[adapter.start() for adapter in self.adapters])
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            await self._shutdown()

    async def _shutdown(self):
        """Graceful shutdown of all adapters."""
        for adapter in self.adapters:
            try:
                await adapter.stop()
            except Exception as e:
                print(f"Error stopping adapter: {e}")


async def run_daemon(config_path: Optional[Path] = None):
    """Main daemon entry point.

    Args:
        config_path: Path to daemon config file
    """
    config = load_daemon_config(config_path)
    gateway = Gateway(config)
    await gateway.start()
