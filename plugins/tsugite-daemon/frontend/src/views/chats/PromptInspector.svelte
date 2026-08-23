<script module lang="ts">
  export interface BreakdownCategory {
    name: string;
    tokens: number;
    /** Per-item detail (per-tool, per-skill…); only its length is surfaced. */
    items?: { name: string; tokens: number }[];
  }
  export interface TokenBreakdown {
    categories: BreakdownCategory[];
    total: number;
  }
</script>

<script lang="ts">
  // The context meter, made into a prompt inspector: clicking it opens a popover
  // with the latest prompt_snapshot's per-category token breakdown. With no
  // breakdown it renders the plain meter unchanged (nothing to open). Popover
  // positioning reuses ModelPicker's clip-boundary flip so it never spills under
  // a scrolling ancestor.
  import { tick } from 'svelte';
  import Meter from '$lib/components/datadisplay/Meter.svelte';
  import { formatTokens } from '$lib/components/chatturns/chatturns.util';
  import { clipBoundaryLeft } from '$lib/dom';
  import { formatAgo } from '$lib/relativeTime';

  let {
    value,
    max,
    label,
    displayText,
    warn = false,
    breakdown,
    turn = null,
    at = null,
    onViewRaw,
  }: {
    value: number;
    max: number;
    label: string;
    displayText: string;
    warn?: boolean;
    breakdown: TokenBreakdown | null;
    /** Turn the snapshot was taken on (0-indexed, as recorded), or null. */
    turn?: number | null;
    /** ISO timestamp of the snapshot, for the staleness readout, or null. */
    at?: string | null;
    /** Opens the raw-messages debug overlay. Omitted -> no such affordance. */
    onViewRaw?: () => void;
  } = $props();

  let root = $state<HTMLElement>();
  let popEl = $state<HTMLElement>();
  let open = $state(false);
  let alignLeft = $state(false);

  // Biggest consumers first; zero-token categories are noise (matches /context).
  const cats = $derived(
    (breakdown?.categories ?? [])
      .filter((c) => c.tokens > 0)
      .slice()
      .sort((a, b) => b.tokens - a.tokens),
  );
  const total = $derived(breakdown?.total ?? 0);
  const windowPct = $derived(max > 0 ? Math.round((total / max) * 100) : 0);
  const share = (n: number): number => (total > 0 ? Math.min(100, (n / total) * 100) : 0);

  // Honest staleness: a replayed/idle snapshot is a past measurement, never
  // live truth. turn is 0-indexed in the log; shown 1-indexed to match bubbles.
  const staleness = $derived.by(() => {
    const parts: string[] = [];
    if (turn != null) parts.push(`turn ${turn + 1}`);
    if (at) {
      const r = formatAgo(at);
      if (r) parts.push(r);
    }
    return parts.length ? `as of ${parts.join(' · ')}` : '';
  });

  async function toggle(): Promise<void> {
    open = !open;
    if (!open) return;
    alignLeft = false;
    await tick();
    if (popEl && popEl.getBoundingClientRect().left < clipBoundaryLeft(popEl) + 8) alignLeft = true;
  }

  $effect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (root && !root.contains(e.target as Node)) open = false;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') open = false;
    };
    window.addEventListener('mousedown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  });
</script>

<div class="ctx-anchor" bind:this={root}>
  {#if breakdown}
    <button
      type="button"
      class="ctx-trigger"
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-label="Show context breakdown"
      onclick={toggle}
    >
      <Meter {value} {max} {label} {displayText} {warn} />
    </button>

    {#if open}
      <div
        class="ctx-pop"
        data-align={alignLeft ? 'left' : 'right'}
        role="dialog"
        aria-label="Context breakdown"
        bind:this={popEl}
      >
        <div class="ctx-hd">
          <div class="ctx-hd-l">
            <span class="ctx-ttl">context breakdown</span>
            {#if staleness}<span class="ctx-stale">{staleness}</span>{/if}
          </div>
          <span class="ctx-total">{formatTokens(total)}</span>
        </div>
        <div class="ctx-rows">
          {#each cats as c (c.name)}
            <div class="ctx-row">
              <span class="ctx-name">{c.name}</span>
              <span class="ctx-bar" class:is-warn={warn}
                ><i style="--w:{share(c.tokens)}%"></i></span
              >
              <span class="ctx-tok">{formatTokens(c.tokens)}</span>
            </div>
          {/each}
        </div>
        <div class="ctx-ft">
          <span>last prompt · ~{windowPct}% of {formatTokens(max)} window</span>
          {#if onViewRaw}
            <button
              type="button"
              class="ctx-raw"
              onclick={() => {
                open = false;
                onViewRaw?.();
              }}>view raw messages</button
            >
          {/if}
        </div>
      </div>
    {/if}
  {:else}
    <Meter {value} {max} {label} {displayText} {warn} />
  {/if}
</div>

<style>
  .ctx-anchor {
    position: relative;
    display: inline-flex;
    flex: none;
  }
  .ctx-trigger {
    display: inline-flex;
    align-items: center;
    background: none;
    border: 0;
    padding: 2px 4px;
    margin: -2px -4px;
    border-radius: var(--r-sm);
    cursor: pointer;
  }
  .ctx-trigger:hover {
    background: var(--bg2);
  }
  .ctx-pop {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 60;
    width: min(300px, 82vw);
    display: flex;
    flex-direction: column;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    overflow: hidden;
  }
  /* Meter near the viewport's left edge: right-anchoring would clip the popover
     off-screen, so it flips to hang rightward instead. */
  .ctx-pop[data-align='left'] {
    right: auto;
    left: 0;
  }
  .ctx-hd {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 8px 11px;
    border-bottom: 1px solid var(--bd0);
  }
  .ctx-hd-l {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
  }
  .ctx-ttl {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .ctx-stale {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    opacity: 0.85;
  }
  .ctx-total {
    font: 600 var(--fs-sm) var(--font-mono);
    color: var(--tx0);
  }
  .ctx-rows {
    display: flex;
    flex-direction: column;
    gap: 5px;
    padding: 9px 11px;
    max-height: 46vh;
    overflow-y: auto;
  }
  .ctx-row {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 56px auto;
    align-items: center;
    gap: 9px;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
  }
  .ctx-name {
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ctx-bar {
    height: 3px;
    background: var(--bg1);
    border-radius: var(--r-full);
    overflow: hidden;
  }
  .ctx-bar i {
    display: block;
    height: 100%;
    width: var(--w, 0%);
    background: var(--tx2);
  }
  .ctx-bar.is-warn i {
    background: var(--st-warn);
  }
  .ctx-tok {
    color: var(--tx2);
    text-align: right;
    white-space: nowrap;
  }
  .ctx-ft {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    padding: 7px 11px;
    border-top: 1px solid var(--bd0);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .ctx-raw {
    flex: none;
    background: none;
    border: 0;
    padding: 0;
    font: inherit;
    color: var(--tx2);
    text-decoration: underline;
    cursor: pointer;
  }
  .ctx-raw:hover {
    color: var(--tx0);
  }
</style>
