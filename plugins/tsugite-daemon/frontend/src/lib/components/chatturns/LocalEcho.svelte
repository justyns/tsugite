<script lang="ts">
  // Ephemeral slash-command echo, Claude-Code style: the command line as the
  // person's own line, its output in a `⎿`-gutter block, a muted footer marking
  // it local-only. Purely presentational - the controller's `localEcho` channel
  // owns the data (never persisted, never sent to the model). Error output reuses
  // the error vocabulary of the timeline's kind:'error' block.
  import Icon from '$lib/components/icon/Icon.svelte';

  let {
    command,
    output,
    ok,
    action,
    onDismiss,
  }: {
    command: string;
    output: string;
    ok: boolean;
    // Optional navigation affordance (e.g. /job's "Open jobs" link).
    action?: { label: string; href: string };
    onDismiss?: () => void;
  } = $props();

  // Empty output must never render an empty gutter block: fall back to a minimal
  // acknowledgment ("done" / "failed") so the command still reads as answered.
  const hasOutput = $derived(output.trim().length > 0);
</script>

<div class="t-echo" class:is-err={!ok}>
  <div class="echo-cmd">
    <span class="who">you</span>
    <span class="arrow">›</span>
    <span class="cmd">{command}</span>
  </div>
  <div class="echo-out" role={ok ? undefined : 'alert'}>
    <span class="gutter" aria-hidden="true">⎿</span>
    {#if hasOutput}
      <pre class="body">{output}</pre>
    {:else}
      <span class="ack">
        <Icon name={ok ? 'check' : 'alert'} size={12} />
        {ok ? 'done' : 'failed'}
      </span>
    {/if}
  </div>
  <div class="echo-foot">
    <span class="echo-note">local only · not saved · not sent to model</span>
    {#if action}
      <a class="echo-act" href={action.href}>{action.label}</a>
    {/if}
    {#if onDismiss}
      <button type="button" class="echo-x" aria-label="Dismiss" onclick={onDismiss}>
        <Icon name="x" size={11} />
      </button>
    {/if}
  </div>
</div>

<style>
  .t-echo {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
    padding: 6px 0 6px 12px;
    border-left: 2px solid var(--bd1);
    font-family: var(--font-mono);
    font-size: var(--fs-xs);
    color: var(--tx2);
  }
  .t-echo.is-err {
    border-left-color: color-mix(in oklab, var(--st-err) 45%, transparent);
  }
  .echo-cmd {
    display: flex;
    align-items: baseline;
    gap: 6px;
    min-width: 0;
  }
  .echo-cmd .who {
    flex: none;
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--acc);
  }
  .echo-cmd .arrow {
    flex: none;
    color: var(--tx3);
  }
  .echo-cmd .cmd {
    min-width: 0;
    color: var(--tx0);
    font-weight: 600;
    overflow-wrap: anywhere;
  }
  .echo-out {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    min-width: 0;
  }
  .echo-out .gutter {
    flex: none;
    color: var(--tx3);
    user-select: none;
    line-height: 1.55;
  }
  .echo-out .body {
    flex: 1;
    min-width: 0;
    margin: 0;
    /* Long output wraps and, if genuinely huge, scrolls inside its own block -
       never widening the pane (the conversation's no-h-scroll rule). */
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    max-height: 320px;
    overflow: auto;
    color: var(--tx1);
    line-height: 1.55;
  }
  .is-err .echo-out .body {
    color: var(--st-err);
  }
  .echo-out .ack {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    color: var(--tx3);
    line-height: 1.55;
  }
  .is-err .echo-out .ack {
    color: var(--st-err);
  }
  .echo-foot {
    display: flex;
    align-items: baseline;
    gap: 10px;
    color: var(--tx3);
    font-size: var(--fs-2xs);
    opacity: 0.85;
  }
  .echo-note {
    min-width: 0;
    margin-right: auto;
  }
  .echo-act {
    flex: none;
    color: var(--acc);
    text-decoration: none;
    opacity: 1;
  }
  .echo-act:hover {
    text-decoration: underline;
  }
  .echo-x {
    flex: none;
    align-self: center;
    display: inline-flex;
    align-items: center;
    padding: 0 2px;
    background: none;
    border: 0;
    border-radius: var(--r-sm);
    color: var(--tx3);
    cursor: pointer;
  }
  .echo-x:hover {
    color: var(--tx0);
    background: var(--bg3);
  }
</style>
