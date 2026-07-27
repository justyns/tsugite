<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import { splitCommand } from './chatturns.util';

  let {
    command,
    status = 'done',
    exitCode,
    meta,
    output,
    args,
    open = false,
    onOpenExternal,
  }: {
    // Full command line; the program token is bolded, the rest shown plain.
    command: string;
    // `ended` = the turn closed before a result arrived (closed-neutral, no exit).
    status?: 'running' | 'done' | 'error' | 'ended';
    // Process exit code for done/error (defaults: done->0, error->1).
    exitCode?: number;
    // Duration ("0.4s") for finished runs, or live elapsed ("12:34").
    meta?: string;
    // Captured stdout/stderr; the collapsible body is omitted when empty.
    output?: string;
    // Native tool-call arguments, shown as a key/value list above the output.
    args?: Record<string, unknown>;
    // Initial expanded state.
    open?: boolean;
    // When provided, a header affordance to open this run elsewhere (e.g. a PTY
    // in the Terminals view).
    onOpenExternal?: () => void;
  } = $props();

  // Follow the `open` prop across status transitions (a running block arrives
  // expanded, then auto-collapses when the run finishes); a manual toggle wins
  // only until the prop next changes.
  let userOverride = $state<{ prop: boolean; value: boolean } | null>(null);
  const isOpen = $derived(userOverride?.prop === open ? userOverride.value : open);

  const parts = $derived(splitCommand(command));

  // At-a-glance args in the title bar: the primary argument's value (tool-aware
  // key order), with a +N tail for the rest; anything long truncates.
  const PRIMARY_ARG_KEYS = [
    'cmd',
    'command',
    'path',
    'url',
    'query',
    'skill_name',
    'name',
    'key',
    'message',
    'prompt',
    'value',
    'text',
  ];
  const ARGS_PREVIEW_MAX = 80;
  const argsPreview = $derived.by(() => {
    if (!args) return '';
    const entries = Object.entries(args);
    if (entries.length === 0) return '';
    const fmt = (v: unknown) => (typeof v === 'string' ? v : JSON.stringify(v));
    const primary = PRIMARY_ARG_KEYS.find((k) => k in args);
    const text = primary
      ? fmt(args[primary]) + (entries.length > 1 ? `  +${entries.length - 1}` : '')
      : entries.map(([k, v]) => `${k}=${fmt(v)}`).join(' ');
    const flat = text.replace(/\s+/g, ' ').trim();
    return flat.length > ARGS_PREVIEW_MAX ? flat.slice(0, ARGS_PREVIEW_MAX - 1) + '…' : flat;
  });
  const exit = $derived(exitCode ?? (status === 'error' ? 1 : 0));
  const argRows = $derived(
    args
      ? Object.entries(args).map(([k, v]) => [k, typeof v === 'string' ? v : JSON.stringify(v)])
      : [],
  );

  function toggle() {
    userOverride = { prop: open, value: !isOpen };
  }
</script>

<div class="t-exec" class:is-open={isOpen} class:is-run={status === 'running'}>
  <!-- A real <button> owns the disclosure (native Enter/Space), and the optional
       external-open control is a sibling — never nested inside another button. -->
  <div class="t-exec-hd">
    <button type="button" class="hd-main" aria-expanded={isOpen} onclick={toggle}>
      <span class="chev"><Icon name="chev-r" size={10} /></span>
      <Icon name="term" size={13} color={status === 'running' ? 'var(--st-ok)' : 'var(--tx3)'} />
      <span class="cmd"><b>{parts.program}</b>{parts.rest}</span>
      <span class="argspv">{argsPreview}</span>
      {#if status === 'running'}
        <span><span class="t-pill" data-st="running"><Spin />running</span></span>
      {:else if status === 'ended'}
        <span class="exit neutral">ended</span>
      {:else}
        <span class="exit {status === 'error' ? 'bad' : 'ok'}">exit {exit}</span>
      {/if}
      {#if meta}<span class="meta">{meta}</span>{/if}
    </button>
    {#if onOpenExternal}
      <button
        type="button"
        class="t-iconbtn open"
        aria-label="Open in Terminals"
        onclick={onOpenExternal}
      >
        <Icon name="out" size={11} />open
      </button>
    {/if}
  </div>
  {#if argRows.length > 0 || output}
    <div class="t-exec-out">
      <div>
        {#if argRows.length > 0}
          <dl class="t-exec-args">
            {#each argRows as [k, v] (k)}
              <div>
                <dt>{k}</dt>
                <dd>{v}</dd>
              </div>
            {/each}
          </dl>
        {/if}
        {#if output}<pre>{output}</pre>{/if}
      </div>
    </div>
  {/if}
</div>

<style>
  .t-exec {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg1);
    font-family: var(--font-mono);
    font-size: var(--fs-sm);
    overflow: hidden;
  }
  .t-exec.is-run {
    border-color: color-mix(in oklab, var(--st-ok) 40%, transparent);
  }
  .t-exec-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 6px 9px;
    min-width: 0;
  }
  .t-exec-hd:hover {
    background: var(--bg2);
  }
  .hd-main {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
    min-width: 0;
    margin: 0;
    padding: 0;
    background: none;
    border: 0;
    color: inherit;
    font: inherit;
    text-align: left;
    cursor: pointer;
    user-select: none;
  }
  .t-exec-hd .chev {
    color: var(--tx3);
    transition: rotate var(--t-2) var(--ease);
    display: inline-flex;
  }
  .t-exec.is-open .t-exec-hd .chev {
    rotate: 90deg;
  }
  .t-exec-hd .cmd {
    color: var(--tx1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 0 1 auto;
    min-width: 4ch;
  }
  /* Args-at-a-glance; doubles as the spacer that pushes the status right. */
  .t-exec-hd .argspv {
    flex: 1 1 auto;
    min-width: 0;
    color: var(--tx3);
    font-size: var(--fs-xs);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-exec-hd .cmd b {
    color: var(--tx0);
    font-weight: 600;
  }
  .t-exec-hd .meta {
    color: var(--tx3);
    font-size: var(--fs-2xs);
    white-space: nowrap;
    font-variant-numeric: tabular-nums;
  }
  .t-exec .exit {
    font-size: var(--fs-2xs);
    font-weight: 600;
    padding: 1px 5px;
    border-radius: 3px;
  }
  .t-exec .exit.ok {
    color: var(--st-ok);
    background: color-mix(in oklab, var(--st-ok) 14%, transparent);
  }
  .t-exec .exit.bad {
    color: var(--st-err);
    background: color-mix(in oklab, var(--st-err) 14%, transparent);
  }
  .t-exec .exit.neutral {
    color: var(--tx3);
    background: var(--bg3);
  }
  .t-exec-out {
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows var(--t-3) var(--ease);
  }
  .t-exec.is-open .t-exec-out {
    grid-template-rows: 1fr;
  }
  .t-exec-out > div {
    overflow: hidden;
  }
  .t-exec-out pre {
    margin: 0;
    padding: 8px 11px;
    border-top: 1px solid var(--bd0);
    background: var(--bg0);
    color: var(--tx2);
    font-size: var(--fs-xs);
    line-height: 1.6;
    max-height: 180px;
    overflow: auto;
  }
  /* Native tool-call arguments: a key/value list above the captured output. */
  .t-exec-args {
    margin: 0;
    padding: 6px 11px;
    border-top: 1px solid var(--bd0);
    display: grid;
    gap: 3px;
    font-size: var(--fs-xs);
  }
  .t-exec-args > div {
    display: flex;
    gap: 8px;
    min-width: 0;
  }
  .t-exec-args dt {
    color: var(--tx3);
    flex: none;
    min-width: 7ch;
  }
  .t-exec-args dd {
    margin: 0;
    color: var(--tx1);
    white-space: pre-wrap;
    word-break: break-word;
    min-width: 0;
  }

  /* Neither maps to a shared component: .t-pill is the exec status pill
     (running/ok/err vocab, unlike Pill's idle/busy/streaming session states),
     and .t-iconbtn is a bare icon-button (no owner; its 11px icon sizing is
     global in tokens.css). */
  .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .t-pill[data-st='running'] {
    --c: var(--st-ok);
  }
  .t-iconbtn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: 0;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: var(--r-sm);
  }
  .t-iconbtn:hover {
    color: var(--tx0);
    background: var(--bg3);
  }
  .t-iconbtn.open {
    align-self: center;
  }
  @media (prefers-reduced-motion: reduce) {
    .t-exec-hd .chev,
    .t-exec-out {
      transition: none;
    }
  }
</style>
