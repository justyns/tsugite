"""Metrics display for multi-step runs."""

from types import SimpleNamespace

from rich.console import Console

from tsugite.agent_runner.metrics import StepMetrics, display_step_metrics


def _logger(console):
    events = []
    return SimpleNamespace(console=console, ui_handler=SimpleNamespace(handle_event=events.append)), events


def test_metrics_table_leaves_the_caller_console_writing_where_it_was():
    """The table is captured into an event, and the console it borrows to render
    goes on writing where it did before - it is the live one the UI prints through."""
    console = Console()
    logger, events = _logger(console)
    before = console.file

    display_step_metrics([StepMetrics(step_name="build", step_number=1, duration=0.5)], logger)

    assert console.file is before
    assert len(events) == 1
    assert "build" in events[0].message


def test_step_plan_preview_leaves_the_caller_console_writing_where_it_was(tmp_path):
    """Same borrow, same contract, on the plan table `tsu render` prints."""
    from tsugite.agent_runner.runner import preview_multistep_agent

    agent = tmp_path / "plan.md"
    agent.write_text(
        "---\nname: plan\nmodel: openai:gpt-4o-mini\n---\n"
        '<!-- tsu:step name="one" -->\nfirst\n\n'
        '<!-- tsu:step name="two" -->\nsecond\n'
    )
    console = Console()
    logger, events = _logger(console)
    before = console.file

    preview_multistep_agent(agent, "task", custom_logger=logger)

    assert console.file is before
    assert any("one" in event.message for event in events)


def test_step_plan_preview_shows_toolsets_without_the_at_sigil(tmp_path):
    from tsugite.agent_runner.runner import preview_multistep_agent

    agent = tmp_path / "plan.md"
    agent.write_text(
        "---\nname: plan\nmodel: openai:gpt-4o-mini\ntools: ['@fs']\n---\n<!-- tsu:step name=\"one\" -->\nfirst\n"
    )
    console = Console()
    logger, events = _logger(console)

    preview_multistep_agent(agent, "task", custom_logger=logger)

    messages = [event.message for event in events]
    assert not any("@fs" in message for message in messages)
    assert any("fs (toolset)" in message for message in messages)
