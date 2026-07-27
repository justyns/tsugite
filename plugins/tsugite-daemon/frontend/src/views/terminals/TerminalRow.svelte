<script lang="ts">
  // Terminal rail row. Reuses the `.t-srow` (mono title, per
  // `.term-list .t-srow .ttl`) and is a mux drag source: dragging it writes a
  // {kind:'terminal'} SurfaceRef so it can be docked into a pane.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { startSpin } from '$lib/components/buttons/spin';
  import { formatElapsed } from '$lib/components/feedback/format';
  import { writeSurfaceDrag } from '$lib/shell/mux/drag';
  import type { Terminal, TerminalState } from '$lib/stores/terminals.svelte';
  import { elapsedSeconds, terminalIndicator, terminalTabState } from './termState';

  let {
    term,
    // Named `st`, not `state` (avoids the `$state` rune / local-binding clash).
    st,
    now,
    isActive = false,
    onSelect,
  }: {
    term: Terminal;
    /** Resolved live state (store overlay wins over the record's own field). */
    st: TerminalState;
    /** Shared wall-clock tick (ms) so every row's elapsed advances together. */
    now: number;
    isActive?: boolean;
    onSelect?: () => void;
  } = $props();

  const ind = $derived(terminalIndicator(st));
  const elapsed = $derived(formatElapsed(elapsedSeconds(term.created_at, term.resolved_at, now)));
  const preview = $derived(term.last_line?.trim() || 'no output yet');
  const attn = $derived(st === 'stream_lost');

  let frame = $state('⠋');
  $effect(() => {
    if (!ind.spin) return;
    return startSpin((glyph) => (frame = glyph));
  });

  function onDragStart(e: DragEvent) {
    if (!e.dataTransfer) return;
    writeSurfaceDrag(e.dataTransfer, {
      kind: 'terminal',
      params: { id: term.id },
      title: term.cmd,
      state: terminalTabState(st),
    });
  }

  function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect?.();
    }
  }
</script>

<div
  class="t-srow"
  class:is-active={isActive}
  class:is-attn={attn}
  role="option"
  tabindex="0"
  aria-selected={isActive}
  aria-label={`${term.cmd} — ${st}`}
  draggable="true"
  ondragstart={onDragStart}
  onclick={() => onSelect?.()}
  onkeydown={onKeydown}
>
  <span class="ind" data-st={st}>
    {#if ind.spin}
      <span class="t-spin" aria-hidden="true">{frame}</span>
    {:else}
      <Icon name={ind.icon} />
    {/if}
  </span>
  <span class="ttl">{term.cmd}</span><span class="when">{elapsed}</span>
  <span class="sub">
    <span class="desc">{preview}</span>
    <span class="mk mono">{term.lines_out} ln</span>
  </span>
</div>

<style>
  /* .t-srow (sidebar row) + `.term-list .t-srow .ttl` mono override. */
  .t-srow {
    display: grid;
    grid-template-columns: 14px 1fr auto;
    grid-template-rows: auto auto;
    gap: 1px 8px;
    padding: 6px 10px 7px;
    border-left: 2px solid transparent;
    cursor: pointer;
    min-width: 0;
    position: relative;
  }
  .t-srow:hover {
    background: var(--bg2);
  }
  .t-srow.is-active {
    background: var(--bg2);
    border-left-color: var(--acc);
  }
  .t-srow.is-attn {
    border-left-color: var(--st-warn);
  }
  .t-srow:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .ind {
    grid-row: 1 / 3;
    display: flex;
    align-items: flex-start;
    padding-top: 4px;
    justify-content: center;
  }
  /* Per-state indicator tint (spinner + icon inherit currentColor). */
  .ind[data-st='running'] {
    color: var(--st-ok);
  }
  .ind[data-st='starting'] {
    color: var(--st-queue);
  }
  .ind[data-st='stream_lost'] {
    color: var(--st-warn);
  }
  .ind[data-st='failed'] {
    color: var(--st-err);
  }
  .ind[data-st='cancelled'],
  .ind[data-st='succeeded'] {
    color: var(--tx3);
  }
  .ttl {
    font: 500 var(--fs-sm) / 1.3 var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .when {
    font: 500 var(--fs-2xs) / 1.5 var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }
  .sub {
    grid-column: 2 / 4;
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
  }
  .sub .desc {
    font-size: var(--fs-xs);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    flex: 1;
    min-width: 0;
  }
  .mk {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    color: var(--tx3);
    flex: none;
    font-size: var(--fs-2xs);
  }
  .t-spin {
    font-family: var(--font-mono);
    font-weight: 600;
    display: inline-block;
    width: 1.1ch;
    line-height: 1;
    flex: none;
  }
</style>
