"""Multi-step agent execution.

A step is not a different kind of run - it is one more prompt inside a run. Each
step goes through `_run_unit` with the agent's content narrowed to that step, so
it gets the same preparation pipeline as a single-shot agent while staying blind
to its siblings' instructions. Only a step's `assign=` result crosses between
them; the retry, loop and metrics policy around that lives here.
"""

import asyncio
import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .helpers import (
    clear_multistep_ui_context,
    get_ui_handler,
    print_step_progress,
    set_multistep_ui_context,
)
from .metrics import StepMetrics, display_step_metrics
from .models import AgentSkippedError
from .runner import _get_model_string, _prepare_step, _run_unit, _RunSetup, execute_prefetch

if TYPE_CHECKING:
    from tsugite.events import EventBus

logger = logging.getLogger(__name__)


class ExecutionContext:
    """Namespace for tsugite-provided execution context.

    Provides access to runtime metadata via attribute access (ctx.user_prompt, ctx.tasks, etc.)
    while keeping user-assigned step variables as top-level names in the execution namespace.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"ExecutionContext({attrs})"


def _build_step_error_message(
    error_type: str,
    step_name: str,
    step_number: int,
    total_steps: int,
    errors: List[str],
    available_vars: List[str],
    previous_step: str,
    max_attempts: int,
    debug_tips: List[str],
) -> str:
    """Build detailed error message for step failures."""
    error_lines = [
        "",
        f"Step {error_type}",
        "━" * 60,
        f"Step: {step_name} ({step_number}/{total_steps})",
        f"Previous Step: {previous_step}",
        f"Attempts: {max_attempts}",
        "",
    ]

    # Add variables section (format depends on whether we have any)
    if available_vars:
        var_label = "Context Variables" if "Template" in error_type else "Available Variables"
        error_lines.append(f"{var_label}: {', '.join(available_vars)}")
    else:
        error_lines.append("Available Variables: None")

    error_lines.extend(["", "Errors:"])

    # Add all error attempts
    for idx, err in enumerate(errors, 1):
        error_lines.append(f"  Attempt {idx}: {err}")

    # Add debugging tips
    error_lines.extend(["━" * 60, "", "To debug:"])
    for tip in debug_tips:
        error_lines.append(f"  {tip}")
    error_lines.append("")

    return "\n".join(error_lines)


def _build_injectable_vars(step_context: Dict[str, Any], assigned_vars: Optional[set] = None) -> Dict[str, Any]:
    """Build variables for injection into Python execution namespace.

    Creates a `ctx` object containing everything. Only user-assigned step variables
    (from `assign="varname"`) are exposed at top-level to avoid namespace pollution.

    Args:
        step_context: Full step context dictionary
        assigned_vars: Set of variable names assigned by user via step.assign_var.
                       If None, no variables are exposed at top-level.

    Returns:
        Dictionary with 'ctx' ExecutionContext and user-assigned variables at top-level
    """
    ctx = ExecutionContext(**step_context)

    # Only user-assigned step variables at top-level (not tsugite metadata)
    if assigned_vars:
        top_level = {k: v for k, v in step_context.items() if k in assigned_vars}
    else:
        top_level = {}

    return {"ctx": ctx, **top_level}


def evaluate_loop_condition(expression: str, context: Dict[str, Any]) -> bool:
    """Evaluate a Jinja2 expression or helper as a boolean condition.

    Args:
        expression: Jinja2 template expression or predefined helper name
        context: Template context with tasks, variables, etc.

    Returns:
        Boolean result of condition evaluation

    Raises:
        ValueError: If expression is invalid or evaluation fails
    """
    from jinja2 import Template, TemplateSyntaxError

    try:
        # Wrap expression in {% if %} to get boolean result
        template_str = f"{{% if {expression} %}}true{{% endif %}}"
        template = Template(template_str)
        result = template.render(**context)
        return result.strip() == "true"
    except TemplateSyntaxError as e:
        raise ValueError(f"Invalid loop condition expression '{expression}': {e}") from e
    except Exception as e:
        raise ValueError(f"Error evaluating loop condition '{expression}': {e}") from e


def _attempt_executed_code(exc: BaseException) -> bool:
    """Did the failed attempt run any code? Retrying would re-issue its side effects."""
    steps = getattr(exc, "execution_steps", None) or []
    return any(getattr(s, "code", "") for s in steps)


def _prepare_retry_context(step_context: Dict[str, Any], step: Any, attempt: int, errors: List[str]) -> None:
    """Add retry-specific variables to step context.

    Args:
        step_context: Step context to update
        step: Step configuration
        attempt: Current attempt number (0-indexed)
        errors: List of previous errors
    """
    step_context["is_retry"] = attempt > 0
    step_context["retry_count"] = attempt
    step_context["max_retries"] = step.max_retries
    step_context["last_error"] = errors[-1] if errors else ""
    step_context["all_errors"] = errors


def _show_step_progress_message(
    custom_logger: Any,
    step_header: str,
    attempt: int,
    max_retries: int,
    i: int,
    step_name: str,
    total_steps: int,
    max_attempts: int,
    debug: bool,
    event_bus: Optional["EventBus"],
) -> None:
    """Display step progress message in UI.

    Args:
        custom_logger: Logger for UI updates
        step_header: Formatted step header
        attempt: Current attempt number (0-indexed)
        max_retries: Maximum retries allowed
        i: Step number (1-indexed)
        step_name: Name of the step
        total_steps: Total number of steps
        max_attempts: Total attempts (retries + 1)
        debug: Debug mode flag
        event_bus: Event bus for debug messages
    """
    if not debug:
        set_multistep_ui_context(custom_logger, i, step_name, total_steps)
        if attempt > 0:
            print_step_progress(custom_logger, step_header, f"Retry {attempt}/{max_retries}...", debug, "yellow")
        else:
            print_step_progress(custom_logger, step_header, "Starting...", debug, "cyan")

    if debug and event_bus:
        from tsugite.events import DebugMessageEvent

        if attempt > 0:
            event_bus.emit(
                DebugMessageEvent(
                    message=f"DEBUG: Retrying Step {i}/{total_steps}: {step_name} "
                    f"(Attempt {attempt + 1}/{max_attempts})"
                )
            )
        else:
            event_bus.emit(DebugMessageEvent(message=f"DEBUG: Executing Step {i}/{total_steps}: {step_name}"))


async def _execute_step_with_retries(
    step: Any,
    step_context: Dict[str, Any],
    agent: Any,
    i: int,
    total_steps: int,
    steps: List[Any],
    step_header: str,
    setup: "_RunSetup",
    prompt: str,
    event_bus: Optional["EventBus"] = None,
    assigned_vars: Optional[set] = None,
) -> tuple[str, float]:
    """Execute a step with automatic retries on failure.

    The step runs through `_run_unit` with the agent's content narrowed to this
    step, so it gets the same preparation pipeline as a single-shot agent while
    staying blind to its siblings' instructions.
    """
    exec_options = setup.exec_options
    custom_logger = setup.custom_logger
    debug = exec_options.debug
    max_attempts = step.max_retries + 1

    # Per-step overrides of the run-level setup:
    #  - results bind to template/namespace variables, so they must be plain
    #    strings; AgentExecutionResult is not JSON-serializable and the default
    #    subprocess executor drops such variables with only a log line. Token
    #    accounting belongs to the run as a whole.
    #  - the shared session gets one turn per step, so each records its own step
    #    header rather than replaying the user's prompt N times.
    #  - a provider session is never resumed across steps: steps are isolated by
    #    design, and for session-owning providers (claude_code, ACP) resuming the
    #    same session id is the one channel that would leak step N-1 into step N.
    step_setup = replace(
        setup,
        exec_options=replace(exec_options, return_token_usage=False),
        user_input_for_history=f"[Step {i}/{total_steps}: {step.name}]",
        resume_session=None,
        resume_after_compaction=False,
    )
    errors = []
    step_start_time = time.time()

    def _render_failure(message: str) -> RuntimeError:
        clear_multistep_ui_context(custom_logger)
        return RuntimeError(
            _build_step_error_message(
                error_type="Template Rendering Failed",
                step_name=step.name,
                step_number=i,
                total_steps=total_steps,
                errors=errors + [message],
                available_vars=list(step_context.keys()),
                previous_step=steps[i - 2].name if i > 1 else "None",
                max_attempts=max_attempts,
                debug_tips=[
                    "1. Check for undefined variables in step template",
                    "2. Verify previous steps assigned expected variables",
                    "3. Run with --debug to see full context",
                ],
            )
        )

    for attempt in range(max_attempts):
        # Add retry context variables
        _prepare_retry_context(step_context, step, attempt, errors)

        # Show progress message
        _show_step_progress_message(
            custom_logger,
            step_header,
            attempt,
            step.max_retries,
            i,
            step.name,
            total_steps,
            max_attempts,
            debug,
            event_bus,
        )

        injectable_vars = _build_injectable_vars(step_context, assigned_vars)

        step_agent = replace(agent, content=step.content)

        if step.spawn_agent_path:
            # A spawn step still goes through the preparation pipeline - it needs
            # the same tool directives and template variables as any other step -
            # and only differs in what it does with the rendered prompt.
            from tsugite.tools.agents import spawn_agent

            try:
                prepared = _prepare_step(step_agent, prompt, step_context, step_setup)
            except Exception as e:
                message = f"Template rendering failed: {e}"
                if attempt == max_attempts - 1:
                    raise _render_failure(message)
                errors.append(message)
                if step.retry_delay > 0:
                    await asyncio.sleep(step.retry_delay)
                continue

            coro = asyncio.to_thread(
                spawn_agent,
                agent_path=step.spawn_agent_path,
                prompt=prepared.rendered_prompt,
            )
        else:
            coro = _run_unit(
                step_agent,
                prompt,
                step_context,
                step_setup,
                model_kwargs=step.model_kwargs,
                injectable_vars=injectable_vars,
            )

        if step.timeout:
            coro = asyncio.wait_for(coro, timeout=step.timeout)

        try:
            step_result = await coro

            # Store result in context if assign variable specified
            if step.assign_var:
                step_context[step.assign_var] = step_result
                if assigned_vars is not None:
                    assigned_vars.add(step.assign_var)
                if debug and event_bus:
                    from tsugite.events import DebugMessageEvent

                    event_bus.emit(DebugMessageEvent(message=f"Assigned result to variable: {step.assign_var}"))

            # Show step completion
            if not debug:
                clear_multistep_ui_context(custom_logger)
                print_step_progress(custom_logger, step_header, "Complete", debug, "green")

            # Calculate duration and return
            step_duration = time.time() - step_start_time
            return step_result, step_duration

        except asyncio.TimeoutError:
            error_msg = f"Step timed out after {step.timeout} seconds"
            errors.append(error_msg)
            code_executed_this_attempt = False
        except AgentSkippedError:
            # A run_if guard is a control-flow signal the scheduler reads to mark
            # the run skipped, not a failure to burn retries on.
            clear_multistep_ui_context(custom_logger)
            raise
        except Exception as e:
            error_msg = str(e)
            errors.append(error_msg)
            code_executed_this_attempt = _attempt_executed_code(e)

        if code_executed_this_attempt and attempt < max_attempts - 1:
            clear_multistep_ui_context(custom_logger)
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(
                    WarningEvent(
                        message=(
                            f"Step '{step.name}' failed after executing code; skipping retry "
                            "to avoid re-issuing side effects."
                        )
                    )
                )
            raise RuntimeError(
                f"Step '{step.name}' failed after executing code: {error_msg}. "
                "Retry skipped to avoid duplicate side effects."
            )

        # If not last attempt, handle retry delay and continue
        if attempt < max_attempts - 1:
            if step.retry_delay > 0:
                await asyncio.sleep(step.retry_delay)
            if not debug and event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Step '{step.name}' failed: {error_msg}"))
        else:
            # Last attempt failed
            clear_multistep_ui_context(custom_logger)
            error_msg = _build_step_error_message(
                error_type="Execution Failed",
                step_name=step.name,
                step_number=i,
                total_steps=total_steps,
                errors=errors,
                available_vars=list(_build_injectable_vars(step_context, assigned_vars).keys()),
                previous_step=steps[i - 2].name if i > 1 else "None",
                max_attempts=max_attempts,
                debug_tips=[
                    "1. Run with --debug to see rendered prompts",
                    "2. Check variable values in previous steps",
                    "3. Verify step dependencies are correct",
                ],
            )
            raise RuntimeError(error_msg)

    # Should never reach here, but for type safety
    raise RuntimeError("Unexpected: Retry loop completed without success or raising")


def _should_repeat_step(
    step: Any, step_context: Dict[str, Any], iteration: int, debug: bool, event_bus: Optional["EventBus"] = None
) -> bool:
    """Determine if a step should repeat based on loop conditions.

    Evaluates repeat_while, repeat_until, and max_iterations to decide
    whether the step should execute again.

    Args:
        step: Step configuration with repeat conditions
        step_context: Current step context for condition evaluation
        iteration: Current iteration count (1-indexed)
        debug: Whether debug mode is active
        event_bus: Optional event bus for emitting debug/warning messages

    Returns:
        True if step should repeat, False otherwise
    """
    should_repeat = False

    # Evaluate repeat conditions
    if step.repeat_while:
        try:
            should_repeat = evaluate_loop_condition(step.repeat_while, step_context)
            if debug and event_bus:
                from tsugite.events import DebugMessageEvent

                event_bus.emit(DebugMessageEvent(message=f"Loop condition (while): {should_repeat}"))
        except Exception as e:
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Loop condition evaluation failed: {e}"))
            should_repeat = False

    elif step.repeat_until:
        try:
            condition_met = evaluate_loop_condition(step.repeat_until, step_context)
            should_repeat = not condition_met  # Repeat UNTIL condition is true
            if debug and event_bus:
                from tsugite.events import DebugMessageEvent

                event_bus.emit(
                    DebugMessageEvent(
                        message=f"Loop condition (until): condition_met={condition_met}, repeat={should_repeat}"
                    )
                )
        except Exception as e:
            if event_bus:
                from tsugite.events import WarningEvent

                event_bus.emit(WarningEvent(message=f"Loop condition evaluation failed: {e}"))
            should_repeat = False

    # Safety check: max iterations
    if should_repeat and iteration >= step.max_iterations:
        if event_bus:
            from tsugite.events import WarningEvent

            event_bus.emit(
                WarningEvent(
                    message=f"⚠️  Step '{step.name}' reached max_iterations ({step.max_iterations}). "
                    f'Use max_iterations="N" to increase limit.'
                )
            )
        should_repeat = False

    return should_repeat


async def run_steps(
    agent: Any,
    prompt: str,
    context: Dict[str, Any],
    setup: "_RunSetup",
) -> str:
    """Run a multi-step agent: one `_run_unit` call per step, sharing one session.

    Steps run sequentially in file order. Only `assign`ed variables cross the
    boundary between them; each step gets a fresh agent loop, so a step never
    inherits a sibling's conversation.
    """
    from tsugite.md_agents import extract_step_directives

    exec_options = setup.exec_options
    custom_logger = setup.custom_logger

    try:
        _, steps = extract_step_directives(agent.content)
    except Exception as e:
        raise ValueError(f"Failed to parse step directives: {e}")

    if not steps:
        raise ValueError("No valid step directives found in agent")

    # Validate unique step names
    step_names = [s.name for s in steps]
    if len(step_names) != len(set(step_names)):
        duplicates = [name for name in step_names if step_names.count(name) > 1]
        raise ValueError(f"Duplicate step names found: {', '.join(set(duplicates))}")

    # Pre-flight: resolve every step's spawn_agent_path so unresolved paths
    # fail before any step runs
    from tsugite.tools.agents import resolve_agent_path

    unresolved = [
        (s.name, s.spawn_agent_path)
        for s in steps
        if s.spawn_agent_path and resolve_agent_path(s.spawn_agent_path) is None
    ]
    if unresolved:
        details = "\n".join(f"  - step '{name}': {path}" for name, path in unresolved)
        raise ValueError(f"Step(s) reference unresolvable agent paths:\n{details}")

    # Create event_bus for emitting events throughout multi-step execution
    from tsugite.events import DebugMessageEvent, EventBus, InfoEvent, WarningEvent

    event_bus = EventBus()
    ui_handler = get_ui_handler(custom_logger)
    if ui_handler:
        event_bus.subscribe(ui_handler.handle_event)

    step_context = {**context, "user_prompt": prompt}

    # Prefetch runs once for the whole agent, not once per step. Clearing it from
    # the config the steps are prepared with is what stops `_run_unit` re-running
    # it on every step.
    if agent.config.prefetch:
        try:
            step_context.update(execute_prefetch(agent.config.prefetch, event_bus))
        except Exception as e:
            event_bus.emit(WarningEvent(message=f"Prefetch execution failed: {e}"))
        agent = replace(agent, config=agent.config.model_copy(update={"prefetch": []}))

    # One session for the run: each step records its own user_input/final_result
    # pair into it, so the conversation reads as the staged workflow it is
    # instead of N orphan sessions.
    if setup.conversation_id is None:
        from .history_integration import open_or_create_session

        try:
            session = open_or_create_session(
                agent_path=agent.file_path,
                agent_name=agent.config.name or (agent.file_path.stem if agent.file_path else "agent"),
                model=_get_model_string(exec_options.model_override, agent.config),
                workspace=setup.workspace.name if setup.workspace else None,
            )
            if session is not None:
                setup.conversation_id = session.session_id
        except Exception as e:
            logger.debug("Could not pre-create session for multi-step run: %s", e)

    try:
        # Execute each step sequentially
        final_result = None
        step_metrics: List[StepMetrics] = []
        assigned_vars: set = set()  # Track user-assigned variables for namespace isolation

        for i, step in enumerate(steps, 1):
            # Add step information to context for this step
            step_context["step_number"] = i
            step_context["step_name"] = step.name
            step_context["total_steps"] = len(steps)

            # Loop control: iterate if step has repeat_while or repeat_until
            iteration = 0
            step_is_looping = bool(step.repeat_while or step.repeat_until)

            while True:
                iteration += 1

                # Add iteration context
                step_context["iteration"] = iteration
                step_context["max_iterations"] = step.max_iterations
                step_context["is_looping_step"] = step_is_looping

                # Show step progress (unless in debug mode which has its own output)
                if step_is_looping:
                    step_header = f"[Step {i}/{len(steps)}: {step.name} (Iteration {iteration})]"
                else:
                    step_header = f"[Step {i}/{len(steps)}: {step.name}]"

                # Execute step with automatic retries
                step_start_time = time.time()
                try:
                    step_result, step_duration = await _execute_step_with_retries(
                        step=step,
                        step_context=step_context,
                        agent=agent,
                        i=i,
                        total_steps=len(steps),
                        steps=steps,
                        step_header=step_header,
                        setup=setup,
                        prompt=prompt,
                        event_bus=event_bus,
                        assigned_vars=assigned_vars,
                    )

                    # Success - store result and record metrics
                    final_result = step_result
                    step_metrics.append(
                        StepMetrics(
                            step_name=step.name,
                            step_number=i,
                            duration=step_duration,
                            status="success",
                        )
                    )

                except RuntimeError as e:
                    # Step execution failed after all retries
                    if step.continue_on_error:
                        # Log warning but continue execution
                        clear_multistep_ui_context(custom_logger)

                        warning_msg = f"⚠ Step '{step.name}' failed but continuing (continue_on_error=true)"
                        event_bus.emit(WarningEvent(message=warning_msg))
                        event_bus.emit(InfoEvent(message=f"Error: {str(e)}"))

                        # Assign None to the variable if specified
                        if step.assign_var:
                            step_context[step.assign_var] = None
                            assigned_vars.add(step.assign_var)
                            if exec_options.debug:
                                event_bus.emit(
                                    DebugMessageEvent(message=f"Assigned None to variable: {step.assign_var}")
                                )

                        # Record metrics for skipped step
                        step_duration = time.time() - step_start_time
                        step_metrics.append(
                            StepMetrics(
                                step_name=step.name,
                                step_number=i,
                                duration=step_duration,
                                status="skipped",
                                error=str(e),
                            )
                        )
                    else:
                        # Re-raise if not continuing on error
                        raise

                # End of step execution - now check if we should repeat the step

                # Check if we should repeat this step (loop control)
                should_repeat = _should_repeat_step(step, step_context, iteration, exec_options.debug, event_bus)

                # Exit while loop if we shouldn't repeat
                if not should_repeat:
                    if step_is_looping and iteration > 1 and not exec_options.debug:
                        event_bus.emit(InfoEvent(message=f"Step '{step.name}' completed after {iteration} iterations"))
                    break

                if not exec_options.debug:
                    event_bus.emit(InfoEvent(message=f"🔁 Repeating step '{step.name}' (iteration {iteration + 1})"))

            # End of while True loop for step iteration

        # Display metrics summary
        if step_metrics:
            display_step_metrics(step_metrics, custom_logger if custom_logger else None)

        return final_result or ""
    finally:
        clear_multistep_ui_context(custom_logger)
