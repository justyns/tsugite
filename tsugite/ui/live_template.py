"""Live UI handler template using Rich Live Display, Tree, and Prompt.

This is a TEMPLATE showing how to create a custom UI mode with:
- Live Display for dynamic updates
- Tree for showing execution hierarchy
- Prompt for interactive elements

To use this template:
1. Rename this file (e.g., live_interactive.py)
2. Rename the class (e.g., LiveInteractiveHandler)
3. Customize the rendering methods for your needs
4. Add helper function in ui/helpers.py
5. Export in ui/__init__.py
6. Integrate in cli/__init__.py

See CLAUDE.md for full UI system documentation.
"""

from contextlib import contextmanager
from typing import Any, Dict, Generator, Optional

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.tree import Tree

from tsugite.ui.base import CustomUIHandler


class LiveTemplateHandler(CustomUIHandler):
    """Template UI handler using Rich Live Display with Tree and interactive prompts.

    Features demonstrated:
    - Live Display: Real-time updates without scrolling
    - Tree: Hierarchical view of execution steps
    - Prompt: Interactive user input during execution
    - Layout: Multi-panel interface

    Customize this to fit your needs!
    """

    def __init__(self, console: Console = None, interactive: bool = True):
        """Initialize the live template handler.

        Args:
            console: Rich console instance (creates default if None)
            interactive: Whether to enable interactive prompts
        """
        if console is None:
            console = Console()

        # Initialize parent with your preferred flags
        super().__init__(
            console=console,
            show_code=True,  # Customize these
            show_observations=True,
            show_llm_messages=False,
            show_execution_results=True,
            show_execution_logs=True,
            show_panels=False,  # We'll use Live Display instead
        )

        # Live display components
        self.live_display: Optional[Live] = None
        self.execution_tree: Optional[Tree] = None
        self.interactive = interactive

        # Track current state for live updates
        self.current_status = "Initializing..."
        self.step_count = 0
        self.tool_calls = []
        self.errors = []

    @contextmanager
    def progress_context(self) -> Generator[None, None, None]:
        """Context manager for Live Display during execution.

        This replaces the default progress spinner with a full Live Display.
        """
        # Create the layout
        layout = self._create_layout()

        # Create Live Display
        self.live_display = Live(
            layout,
            console=self.console,
            refresh_per_second=4,  # Adjust refresh rate
            screen=False,  # Set to True for full-screen mode
        )

        with self.live_display:
            try:
                yield
            finally:
                self.live_display = None

    def _create_layout(self) -> Layout:
        """Create the layout structure for Live Display.

        Customize this to arrange your panels however you want.

        Returns:
            Layout with configured panels
        """
        # Create main layout with multiple sections
        layout = Layout()

        # Split into header, body, footer
        layout.split(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )

        # Split body into left (tree) and right (details)
        layout["body"].split_row(
            Layout(name="tree", ratio=1),
            Layout(name="details", ratio=2),
        )

        # Initial content
        layout["header"].update(self._render_header())
        layout["tree"].update(self._render_tree())
        layout["details"].update(self._render_details())
        layout["footer"].update(self._render_footer())

        return layout

    def _render_header(self) -> Panel:
        """Render the header panel.

        Returns:
            Panel with header content
        """
        return Panel(
            f"[bold cyan]Tsugite Agent Execution[/bold cyan] | Status: {self.current_status}",
            style="cyan",
        )

    def _render_tree(self) -> Panel:
        """Render the execution tree.

        This shows the hierarchical structure of execution steps.

        Returns:
            Panel with tree content
        """
        if self.execution_tree is None:
            # Create initial tree
            self.execution_tree = Tree("🚀 [bold]Execution", guide_style="dim")

        return Panel(self.execution_tree, title="[bold]Execution Tree[/bold]", border_style="green")

    def _render_details(self) -> Panel:
        """Render the details panel.

        Shows current step information, code, observations, etc.

        Returns:
            Panel with detail content
        """
        # Build details from current state
        details = []

        if self.state.task:
            details.append(f"[bold]Task:[/bold] {self.state.task}")
            details.append("")

        if self.state.current_step:
            details.append(f"[bold]Step:[/bold] {self.state.current_step}")

        if self.state.code_being_executed:
            details.append("\n[bold yellow]Executing Code:[/bold yellow]")
            # Truncate long code
            code_preview = self.state.code_being_executed[:200]
            if len(self.state.code_being_executed) > 200:
                code_preview += "..."
            details.append(f"[dim]{code_preview}[/dim]")

        # Show recent tool calls
        if self.tool_calls:
            details.append("\n[bold]Recent Tools:[/bold]")
            for tool in self.tool_calls[-3:]:  # Last 3 tools
                details.append(f"  • {tool}")

        # Show errors if any
        if self.errors:
            details.append("\n[bold red]Errors:[/bold red]")
            for error in self.errors[-2:]:  # Last 2 errors
                details.append(f"  ⚠️  {error}")

        content = "\n".join(details) if details else "[dim]No details yet...[/dim]"
        return Panel(content, title="[bold]Details[/bold]", border_style="blue")

    def _render_footer(self) -> Panel:
        """Render the footer panel.

        Shows summary statistics, costs, etc.

        Returns:
            Panel with footer content
        """
        stats = f"Steps: {self.step_count} | Tools: {len(self.tool_calls)} | Errors: {len(self.errors)}"
        return Panel(stats, style="dim")

    def _update_live_display(self):
        """Update the live display with current state.

        Call this whenever you want to refresh the display.
        """
        if self.live_display and self.live_display.is_started:
            layout = self._create_layout()
            self.live_display.update(layout)

    # ============================================================================
    # Event Handlers - Override these to customize behavior
    # ============================================================================

    def _handle_task_start(self, data: Dict[str, Any]) -> None:
        """Handle task start event."""
        self.state.task = data.get("task")
        self.current_status = "Starting..."

        # Create root of execution tree
        self.execution_tree = Tree(
            f"🚀 [bold]{self.state.task[:50]}...[/bold]"
            if len(self.state.task or "") > 50
            else f"🚀 [bold]{self.state.task}[/bold]",
            guide_style="dim",
        )

        self._update_live_display()

    def _handle_step_start(self, data: Dict[str, Any]) -> None:
        """Handle step start event."""
        self.state.current_step = data.get("step", self.state.current_step + 1)
        self.step_count = self.state.current_step
        self.current_status = f"Step {self.state.current_step}: Thinking..."

        # Add step to tree
        if self.execution_tree:
            step_label = (
                f"Round {self.state.current_step}"
                if self.state.multistep_context
                else f"Step {self.state.current_step}"
            )
            self.execution_tree.add(f"🤔 {step_label}", style="cyan")

        self._update_live_display()

    def _handle_code_execution(self, data: Dict[str, Any]) -> None:
        """Handle code execution event."""
        self.state.code_being_executed = data.get("code")
        self.current_status = f"Step {self.state.current_step}: Executing code..."

        # Add code execution to tree
        if self.execution_tree and self.execution_tree.children:
            last_step = self.execution_tree.children[-1]
            code_preview = (
                self.state.code_being_executed[:40] + "..."
                if len(self.state.code_being_executed) > 40
                else self.state.code_being_executed
            )
            last_step.add(f"⚡ Code: {code_preview}", style="yellow")

        self._update_live_display()

    def _handle_tool_call(self, data: Dict[str, Any]) -> None:
        """Handle tool call event."""
        content = data.get("content", "")

        # Extract tool name from content
        if "Calling tool:" in content:
            tool_name = content.split("Calling tool:")[1].split("with")[0].strip()
            self.tool_calls.append(tool_name)
            self.current_status = f"Step {self.state.current_step}: Using {tool_name}..."

            # Add to tree
            if self.execution_tree and self.execution_tree.children:
                last_step = self.execution_tree.children[-1]
                last_step.add(f"🔧 {tool_name}", style="magenta")

        self._update_live_display()

    def _handle_observation(self, data: Dict[str, Any]) -> None:
        """Handle observation event."""
        observation = data.get("observation", "")
        self.current_status = f"Step {self.state.current_step}: Processing results..."

        # Check for errors in observation
        is_error = any(keyword in observation.lower() for keyword in ["error", "failed", "exception", "traceback"])

        if is_error:
            self.errors.append(observation[:100])
            # Add error to tree
            if self.execution_tree and self.execution_tree.children:
                last_step = self.execution_tree.children[-1]
                last_step.add("❌ Error", style="red")
        else:
            # Add success to tree
            if self.execution_tree and self.execution_tree.children:
                last_step = self.execution_tree.children[-1]
                obs_preview = observation[:40] + "..." if len(observation) > 40 else observation
                last_step.add(f"✓ {obs_preview}", style="green")

        self._update_live_display()

    def _handle_final_answer(self, data: Dict[str, Any]) -> None:
        """Handle final answer event."""
        answer = data.get("answer", "")
        self.current_status = "✅ Complete"

        # Add final answer to tree
        if self.execution_tree:
            self.execution_tree.add(f"✅ [bold green]Final Answer: {answer[:50]}...[/bold green]")

        self._update_live_display()

        # Optional: Show interactive confirmation
        if self.interactive:
            self.live_display.stop()
            self.console.print("\n" + "=" * 60)
            self.console.print(f"[bold green]Final Answer:[/bold green] {answer}")
            self.console.print("=" * 60)

            # Example of using Prompt
            # satisfied = Confirm.ask("Are you satisfied with this answer?")
            # if not satisfied:
            #     # Could trigger retry or refinement
            #     pass

    def _handle_error(self, data: Dict[str, Any]) -> None:
        """Handle error event."""
        error = data.get("error", "")
        error_type = data.get("error_type", "Error")

        self.errors.append(f"{error_type}: {error[:100]}")
        self.current_status = f"❌ {error_type}"

        # Add error to tree
        if self.execution_tree:
            self.execution_tree.add(f"❌ [bold red]{error_type}[/bold red]", style="red")

        self._update_live_display()

    def _handle_reasoning_content(self, data: Dict[str, Any]) -> None:
        """Handle reasoning content from models like Claude."""
        content = data.get("content", "")

        # Add reasoning to tree
        if self.execution_tree and content:
            reasoning_preview = content[:50] + "..." if len(content) > 50 else content
            self.execution_tree.add(f"🧠 Reasoning: {reasoning_preview}", style="magenta")

        self._update_live_display()

    def _handle_cost_summary(self, data: Dict[str, Any]) -> None:
        """Handle cost summary display."""
        cost = data.get("cost")
        total_tokens = data.get("total_tokens")

        # Update footer with cost info
        if cost or total_tokens:
            # Could update footer or show in details panel
            # Example: Add cost/token info to self.current_status or track in instance variable
            pass

        self._update_live_display()

    # ============================================================================
    # Interactive Prompt Examples
    # ============================================================================

    def prompt_for_confirmation(self, message: str) -> bool:
        """Example: Prompt user for confirmation.

        Args:
            message: Confirmation message

        Returns:
            True if confirmed, False otherwise
        """
        if not self.interactive:
            return True

        # Pause live display for interaction
        if self.live_display:
            self.live_display.stop()

        result = Confirm.ask(message)

        # Resume live display
        if self.live_display:
            self.live_display.start()

        return result

    def prompt_for_input(self, message: str, default: str = "") -> str:
        """Example: Prompt user for text input.

        Args:
            message: Prompt message
            default: Default value

        Returns:
            User input
        """
        if not self.interactive:
            return default

        # Pause live display for interaction
        if self.live_display:
            self.live_display.stop()

        result = Prompt.ask(message, default=default)

        # Resume live display
        if self.live_display:
            self.live_display.start()

        return result


# ============================================================================
# Example: Alternative implementation with Table instead of Tree
# ============================================================================


class LiveTableHandler(LiveTemplateHandler):
    """Alternative template using Table for step tracking.

    Shows how to create variants of the same UI pattern.
    """

    def __init__(self, console: Console = None, interactive: bool = True):
        super().__init__(console, interactive)
        self.steps_table = Table(show_header=True, header_style="bold cyan")
        self.steps_table.add_column("Step", style="cyan", width=8)
        self.steps_table.add_column("Status", width=12)
        self.steps_table.add_column("Action", overflow="fold")

    def _render_tree(self) -> Panel:
        """Override to use Table instead of Tree."""
        return Panel(
            self.steps_table,
            title="[bold]Execution Steps[/bold]",
            border_style="green",
        )

    def _handle_step_start(self, data: Dict[str, Any]) -> None:
        """Add step as table row."""
        step = data.get("step", 1)
        self.steps_table.add_row(
            f"Step {step}",
            "[yellow]Running[/yellow]",
            "Waiting for LLM...",
        )
        self._update_live_display()

    # Override other handlers as needed to update table instead of tree...


# ============================================================================
# Helper function (add this to ui/helpers.py)
# ============================================================================


def create_live_template_logger(interactive: bool = True):
    """Create logger using Live Template handler.

    Args:
        interactive: Enable interactive prompts

    Returns:
        CustomUILogger with LiveTemplateHandler

    Example:
        from tsugite.ui.helpers import create_live_template_logger

        logger = create_live_template_logger(interactive=True)
        # Use with agent runner
    """
    from tsugite.ui.base import CustomUILogger
    # from tsugite.ui.live_template import LiveTemplateHandler  # After renaming

    handler = LiveTemplateHandler(interactive=interactive)
    return CustomUILogger(handler, handler.console)
