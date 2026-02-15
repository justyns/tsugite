"""Initialization command for first-time setup."""

import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from tsugite.config import get_config_path, load_config, save_config

console = Console()


def _get_questionary_style():
    """Lazy load questionary style."""
    from prompt_toolkit.styles import Style

    # Custom questionary style for better readability
    # Uses dark background for popup with light text for good contrast
    return Style(
        [
            ("qmark", "fg:ansicyan bold"),  # Question mark
            ("question", "bold"),  # Question text
            ("answer", "fg:ansicyan bold"),  # Selected answer
            ("pointer", "fg:ansicyan bold"),  # Pointer arrow
            ("highlighted", "fg:ansiwhite bg:ansiblue"),  # Highlighted choice - white on blue
            ("selected", "fg:ansicyan"),  # Selected text
            ("separator", "fg:ansiblack"),  # Separator
            ("instruction", ""),  # Instructions
            ("text", ""),  # Default text
            # Autocomplete-specific styles
            ("completion-menu", "bg:ansiblack fg:ansiwhite"),  # Popup background
            ("completion-menu.completion", "fg:ansiwhite bg:ansiblack"),  # Individual items
            (
                "completion-menu.completion.current",
                "fg:ansiblack bg:ansicyan bold",
            ),  # Current selection - black on cyan
        ]
    )


@contextmanager
def suppress_litellm_warnings():
    """Temporarily suppress LiteLLM warnings."""
    litellm_logger = logging.getLogger("LiteLLM")
    original_level = litellm_logger.level
    litellm_logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        litellm_logger.setLevel(original_level)


def detect_ollama() -> bool:
    """Check if Ollama is running."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=2,
            check=False,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def fetch_provider_models(provider: str) -> List[str]:
    """Get available models for a provider using LiteLLM.

    Args:
        provider: Provider name (ollama, openai, anthropic, google)

    Returns:
        List of model strings in LiteLLM format (e.g., "ollama/qwen2.5-coder:7b")
    """
    try:
        from litellm import get_valid_models

        with suppress_litellm_warnings():
            # For ollama, check endpoint to get locally installed models
            if provider == "ollama":
                models = get_valid_models(
                    custom_llm_provider="ollama",
                    check_provider_endpoint=True,
                )
            # For cloud providers, get models based on API key presence
            elif provider == "openai":
                models = get_valid_models(custom_llm_provider="openai")
            elif provider == "anthropic":
                models = get_valid_models(custom_llm_provider="anthropic")
            elif provider == "google":
                # Use "gemini" instead of "google" for LiteLLM
                models = get_valid_models(custom_llm_provider="gemini")
                # Convert gemini/* to google:* for consistency
                models = [m.replace("gemini/", "google/") for m in models]
            else:
                return []

        return models if models else []
    except Exception:
        # Silently return empty list on error
        return []


def litellm_to_tsugite_format(model: str) -> str:
    """Convert LiteLLM model format to Tsugite format.

    Args:
        model: Model in LiteLLM format (e.g., "ollama/qwen2.5-coder:7b")

    Returns:
        Model in Tsugite format (e.g., "ollama:qwen2.5-coder:7b")
    """
    # Replace first / with :
    return model.replace("/", ":", 1)


def detect_available_providers() -> dict[str, int]:
    """Detect which LLM providers are available and count their models.

    Returns:
        Dict of provider name -> model count (0 if unavailable)
    """
    providers = {}

    # Check each provider
    for provider_name in ["ollama", "openai", "anthropic", "google"]:
        models = fetch_provider_models(provider_name)
        providers[provider_name] = len(models)

    return providers


def prompt_for_model(providers: dict[str, int], skip_prompt: bool = False, default_model: Optional[str] = None) -> str:
    """Prompt user to select a provider and then a model with arrow-key navigation.

    Args:
        providers: Dict of provider name -> model count
        skip_prompt: If True, auto-select first available provider and model
        default_model: If provided, skip prompts entirely

    Returns:
        Model string in Tsugite format (e.g., "ollama:qwen2.5-coder:7b")
    """
    import questionary

    if default_model:
        return default_model

    questionary_style = _get_questionary_style()
    console.print("\n[bold cyan]Model Selection[/bold cyan]")

    # Filter available providers
    provider_choices = [(name, count) for name, count in providers.items() if count > 0]

    if not provider_choices:
        console.print("\n[yellow]No providers detected. You can still configure manually.[/yellow]")
        if skip_prompt:
            return "ollama:qwen2.5-coder:7b"

        model = questionary.text(
            "Enter model string (e.g., 'ollama:qwen2.5-coder:7b', 'openai:gpt-4o-mini'):",
            default="ollama:qwen2.5-coder:7b",
            style=questionary_style,
        ).ask()
        return model or "ollama:qwen2.5-coder:7b"

    # Loop to allow going back
    while True:
        # Step 1: Select Provider
        if skip_prompt:
            selected_provider = provider_choices[0][0]
        else:
            provider_options = [f"{name} ({count} models)" for name, count in provider_choices]

            selected_display = questionary.select(
                "Select a provider:",
                choices=provider_options,
                style=questionary_style,
            ).ask()

            if not selected_display:  # User cancelled (Ctrl+C)
                raise KeyboardInterrupt

            # Extract provider name from display string
            selected_provider = selected_display.split(" (")[0]

        # Step 2: Select Model from Provider
        console.print(f"\n[bold cyan]Fetching {selected_provider} models...[/bold cyan]")
        models = fetch_provider_models(selected_provider)

        if not models:
            console.print(f"[yellow]No models found for {selected_provider}[/yellow]")
            if skip_prompt:
                return f"{selected_provider}:default"

            model = questionary.text(
                f"Enter {selected_provider} model name:",
                default="default",
                style=questionary_style,
            ).ask()
            return f"{selected_provider}:{model or 'default'}"

        # Convert to Tsugite format
        tsugite_models = [litellm_to_tsugite_format(m) for m in models]

        if skip_prompt:
            return tsugite_models[0]

        # Build model choices with Back option at the end
        special_choices = ["← Back to provider selection", "✎ Enter custom model name"]
        model_choices = tsugite_models + special_choices

        # Use autocomplete for searchable/filterable list
        console.print("\n[dim]💡 Tip: Type to filter models, use arrow keys to navigate[/dim]")
        selected_model = questionary.autocomplete(
            f"Select a model from {selected_provider}:",
            choices=model_choices,
            match_middle=True,  # Allow matching in middle of string
            style=questionary_style,
        ).ask()

        if not selected_model:  # User cancelled
            raise KeyboardInterrupt

        if selected_model == "← Back to provider selection":
            # Go back to provider selection
            continue
        elif selected_model == "✎ Enter custom model name":
            # Custom model name
            model_name = questionary.text(
                f"Enter {selected_provider} model name:",
                style=questionary_style,
            ).ask()
            if model_name:
                return f"{selected_provider}:{model_name}"
            else:
                continue  # Go back if empty
        else:
            # Regular model selected
            return selected_model


def test_setup(model: str, skip_test: bool = False) -> bool:
    """Run a quick test to verify setup works."""
    if skip_test:
        return True

    console.print("\n[bold cyan]Testing Setup[/bold cyan]")

    if not Confirm.ask("Run a quick test?", default=True):
        return True

    try:
        from io import StringIO

        from rich.console import Console as RichConsole

        # Use the builtin default agent
        from tsugite.agent_inheritance import find_agent_file
        from tsugite.agent_runner import run_agent
        from tsugite.ui import custom_agent_ui

        agent_path = find_agent_file("default", current_agent_dir=Path.cwd())

        console.print("[dim]Running: tsugite run +default 'say hello'[/dim]")

        # Create a quiet logger that doesn't output anything
        quiet_console = RichConsole(file=StringIO())
        with custom_agent_ui(
            quiet_console,
            show_code=False,
            show_observations=False,
            show_progress=False,
            show_llm_messages=False,
            show_panels=False,
        ) as logger:
            from tsugite.options import ExecutionOptions

            result = run_agent(
                agent_path=agent_path,
                prompt="Say hello in one sentence",
                exec_options=ExecutionOptions(model_override=model),
                custom_logger=logger,
            )

        console.print("[green]✓[/green] Test successful!")
        console.print(
            f"[dim]Response: {result[:100]}...[/dim]" if len(result) > 100 else f"[dim]Response: {result}[/dim]"
        )
        return True

    except Exception as e:
        console.print(f"[yellow]Warning: Test failed: {e}[/yellow]")
        console.print("[dim]You can still use tsugite, but you may need to configure your model.[/dim]")
        return False


def init(
    model: Optional[str] = typer.Option(None, "--model", help="Model to use (skip prompt)"),
    force: bool = typer.Option(False, "--force", help="Overwrite existing configuration"),
):
    """Initialize tsugite with global configuration."""
    from tsugite.cli.helpers import get_logo

    # Show welcome
    console.print(get_logo(console), style="cyan")
    console.print()
    console.print(
        Panel(
            "[bold]Welcome to Tsugite![/bold]\n\n"
            "This wizard will set up your global configuration.\n"
            "You can reconfigure anytime with: [cyan]tsugite config[/cyan]",
            border_style="blue",
            title="First-Time Setup",
        )
    )

    # Check if already configured
    config_path = get_config_path()
    config = load_config()

    if config_path.exists() and config.default_model and not force:
        console.print(f"\n[yellow]Configuration already exists at: {config_path}[/yellow]")
        console.print(f"Current default model: [cyan]{config.default_model}[/cyan]")

        if not Confirm.ask("Reconfigure?", default=False):
            console.print("\n[dim]Setup cancelled. Use --force to overwrite.[/dim]")
            return

    # Detect providers
    console.print("\n[bold cyan]Detecting LLM Providers[/bold cyan]")
    providers = detect_available_providers()

    # Prompt for model
    selected_model = prompt_for_model(providers, skip_prompt=bool(model), default_model=model)

    # Create config
    console.print("\n[bold cyan]Creating Configuration[/bold cyan]")
    config.default_model = selected_model
    save_config(config)

    console.print(f"[green]✓[/green] Configuration saved to: [dim]{config_path}[/dim]")
    console.print(f"[green]✓[/green] Default model: [cyan]{selected_model}[/cyan]")

    # Show next steps
    console.print("\n" + "=" * 60)
    console.print("[bold green]Setup Complete![/bold green]")
    console.print("=" * 60)

    console.print("\n[bold]Next Steps:[/bold]")
    console.print("  1. Run the default agent: [cyan]tsugite run +default 'your task'[/cyan]")
    console.print("  2. View configuration: [cyan]tsugite config show[/cyan]")
    console.print("  3. Explore agents: [cyan]tsugite agents list[/cyan]")
    console.print("  4. Create a workspace: [cyan]tsugite workspace init my-workspace[/cyan]")

    console.print("\n[bold]Documentation:[/bold]")
    console.print("  • Quick start: See README.md")
    console.print("  • Agent guide: See CLAUDE.md")
    console.print("  • Examples: See examples/ directory")

    console.print()
