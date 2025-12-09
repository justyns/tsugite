"""Tab completion for REPL chat mode."""

import os
from pathlib import Path
from typing import Iterable, List

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document

from tsugite.agent_inheritance import get_builtin_agents_path, get_global_agents_paths


class TsugiteCompleter(Completer):
    """Smart tab completion for tsugite REPL.

    Provides context-aware completion for:
    - Slash commands
    - Agent names
    - File paths (for attachments)
    - Model names (from config)
    """

    # Slash commands with descriptions
    COMMANDS = {
        "/help": "Show available commands",
        "/exit": "Exit the REPL",
        "/quit": "Exit the REPL",
        "/clear": "Clear the screen",
        "/model": "Switch to a different model",
        "/agent": "Switch to a different agent",
        "/attach": "Attach a file for context",
        "/detach": "Remove an attachment",
        "/list-attachments": "Show current attachments",
        "/continue": "Resume a previous conversation",
        "/history": "Show recent conversations",
        "/save": "Export conversation to file",
        "/stats": "Show token and cost statistics",
        "/tools": "Show available tools for current agent",
        "/cost": "Show cumulative session cost",
        "/stream": "Toggle streaming mode on/off",
        "/verbose": "Toggle verbose mode (show raw tool output)",
    }

    def __init__(self, current_agent_name: str = ""):
        """Initialize completer.

        Args:
            current_agent_name: Name of currently loaded agent
        """
        self.current_agent_name = current_agent_name
        self.agent_names = self._discover_agents()

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Get completions for the current cursor position.

        Args:
            document: Current document state
            complete_event: Completion event

        Yields:
            Completion objects for matching items
        """
        text = document.text_before_cursor

        # Command completion (line starts with /)
        if text.startswith("/"):
            # Extract command and arguments
            parts = text.split(maxsplit=1)
            command = parts[0]

            # If we're still typing the command (no space after it)
            if len(parts) == 1:
                for cmd, desc in self.COMMANDS.items():
                    if cmd.startswith(command):
                        yield Completion(
                            cmd,
                            start_position=-len(command),
                            display_meta=desc,
                        )
            # Command-specific argument completion
            elif command == "/agent":
                # Complete agent names
                if len(parts) > 1:
                    prefix = parts[1]
                    for agent in self.agent_names:
                        if agent.startswith(prefix):
                            yield Completion(
                                agent,
                                start_position=-len(prefix),
                                display_meta="agent",
                            )
            elif command in ("/attach", "/detach"):
                # Complete file paths
                if len(parts) > 1:
                    prefix = parts[1]
                    for path in self._complete_path(prefix):
                        yield Completion(
                            path,
                            start_position=-len(prefix),
                            display_meta="file",
                        )
            elif command == "/stream":
                # Complete on/off
                if len(parts) > 1:
                    prefix = parts[1].lower()
                    for option in ["on", "off"]:
                        if option.startswith(prefix):
                            yield Completion(
                                option,
                                start_position=-len(prefix),
                            )
            elif command == "/verbose":
                # Complete on/off
                if len(parts) > 1:
                    prefix = parts[1].lower()
                    for option in ["on", "off"]:
                        if option.startswith(prefix):
                            yield Completion(
                                option,
                                start_position=-len(prefix),
                            )
        # No completion for free text (user prompts)
        # This keeps the REPL simple and doesn't interfere with natural language

    def _discover_agents(self) -> List[str]:
        """Discover available agent names.

        Returns:
            List of agent names (without .md extension)
        """
        agents = set()

        # Check current directory
        cwd = Path.cwd()
        for location in [cwd / ".tsugite", cwd / "agents", cwd]:
            if location.exists() and location.is_dir():
                for file in location.glob("*.md"):
                    agents.add(file.stem)

        # Check builtin agents
        builtin_path = get_builtin_agents_path()
        if builtin_path.exists():
            for file in builtin_path.glob("*.md"):
                agents.add(file.stem)

        # Check global agent directories
        for global_dir in get_global_agents_paths():
            if global_dir.exists():
                for file in global_dir.glob("*.md"):
                    agents.add(file.stem)

        return sorted(agents)

    def _complete_path(self, prefix: str) -> List[str]:
        """Complete file paths.

        Args:
            prefix: Partial path to complete

        Returns:
            List of matching paths
        """
        if not prefix:
            prefix = "."

        # Expand user home directory
        prefix = os.path.expanduser(prefix)

        # Get directory and filename parts
        if prefix.endswith("/"):
            directory = prefix
            filename_prefix = ""
        else:
            directory = os.path.dirname(prefix) or "."
            filename_prefix = os.path.basename(prefix)

        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return []

            completions = []
            for item in sorted(dir_path.iterdir()):
                name = item.name
                if name.startswith(filename_prefix):
                    if item.is_dir():
                        completions.append(str(item) + "/")
                    else:
                        completions.append(str(item))

            return completions
        except (OSError, PermissionError):
            return []

    def update_agent_name(self, agent_name: str) -> None:
        """Update the current agent name.

        Args:
            agent_name: New agent name
        """
        self.current_agent_name = agent_name
