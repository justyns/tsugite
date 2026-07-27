<script lang="ts">
  import type { Snippet } from 'svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Chip from '$lib/components/buttons/Chip.svelte';

  // An agent's artifact rendered decoupled from the message flow. `panel` is
  // the docked pane: a header with the rendered/diff/raw/json mode switch, a
  // scrollable body whose active view is chosen by data-view, an optional
  // overlay (selection popover) and footer (verdict bar). `launch` is the
  // compact card that sits in chat and opens the pane.
  // The launch card's .t-btn stays inline (bound up with the component-specific
  // .t-artlaunch layout); the mode switch reuses Seg and the kind tag reuses Chip.
  let {
    variant = 'panel',
    title,
    kind,
    count,
    view = $bindable('rendered'),
    views = ['rendered', 'diff', 'raw', 'json'],
    onViewChange,
    subtitle,
    openLabel = 'Review',
    onOpen,
    rendered,
    diff,
    raw,
    json,
    edit,
    overlay,
    footer,
  }: {
    variant?: 'panel' | 'launch';
    title: string;
    kind?: string;
    count?: string;
    view?: string;
    views?: string[];
    onViewChange?: (view: string) => void;
    subtitle?: string;
    openLabel?: string;
    onOpen?: () => void;
    rendered?: Snippet;
    diff?: Snippet;
    raw?: Snippet;
    json?: Snippet;
    /** editor view - the wiki rendered/raw/edit toggle reuses this frame */
    edit?: Snippet;
    overlay?: Snippet;
    footer?: Snippet;
  } = $props();

  function select(next: string) {
    view = next;
    onViewChange?.(next);
  }
</script>

{#if variant === 'launch'}
  <div class="t-artlaunch">
    <span class="ai">
      <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
        ><path d="M4 2.5h5l3 3v8H4z" /><path d="M9 2.5v3h3" /></svg
      >
    </span>
    <div class="am">
      <span class="t">{title}</span>
      {#if subtitle}<span class="s">{subtitle}</span>{/if}
    </div>
    <button type="button" class="t-btn t-btn--pri t-btn--sm" onclick={() => onOpen?.()}>
      <svg class="ic" viewBox="0 0 16 16" aria-hidden="true"
        ><path d="M6.5 3.5h-3v9h9v-3" /><path d="M9.5 2.5h4v4M13 3L8.2 7.8" /></svg
      >{openLabel}
    </button>
  </div>
{:else}
  <div class="art-panel">
    <div class="art-hd">
      <svg class="ic ic--file" viewBox="0 0 16 16" aria-hidden="true"
        ><path d="M4 2.5h5l3 3v8H4z" /><path d="M9 2.5v3h3" /></svg
      >
      <span class="ttl mono">{title}</span>
      {#if kind}<Chip>{kind}</Chip>{/if}
      <div class="grow"></div>
      {#if count}<span class="cnt mono">{count}</span>{/if}
      <Seg options={views} bind:value={() => view, select} ariaLabel="Artifact view" />
    </div>

    <!-- focusable so keyboard users can scroll the overflowing doc region -->
    <!-- svelte-ignore a11y_no_noninteractive_tabindex -->
    <div class="art-bd" data-view={view} tabindex="0" role="region" aria-label={`${title} content`}>
      <div data-art-view="rendered">{@render rendered?.()}</div>
      <div data-art-view="diff">{@render diff?.()}</div>
      <div data-art-view="raw">{@render raw?.()}</div>
      <div data-art-view="json">{@render json?.()}</div>
      <div data-art-view="edit">{@render edit?.()}</div>
    </div>

    {@render overlay?.()}
    {@render footer?.()}
  </div>
{/if}

<style>
  .art-panel {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    overflow: hidden;
    position: relative;
  }
  .art-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 11px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg2);
    flex-wrap: wrap;
  }
  .art-hd .ttl {
    font: 600 var(--fs-sm) / 1.2 var(--font-mono);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
    color: var(--tx0);
  }
  .art-hd .ic--file {
    color: var(--acc);
  }
  .cnt {
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .mono {
    font-family: var(--font-mono);
  }

  .art-bd {
    flex: 1;
    overflow-y: auto;
    position: relative;
    padding: 14px 16px;
    min-width: 0;
    min-height: 0;
  }
  .art-bd[data-view='raw'],
  .art-bd[data-view='json'] {
    padding: 0;
  }
  .art-bd > [data-art-view] {
    display: none;
  }
  .art-bd[data-view='rendered'] > [data-art-view='rendered'],
  .art-bd[data-view='diff'] > [data-art-view='diff'],
  .art-bd[data-view='raw'] > [data-art-view='raw'],
  .art-bd[data-view='json'] > [data-art-view='json'],
  .art-bd[data-view='edit'] > [data-art-view='edit'] {
    display: block;
  }

  /* content styling reaches snippet markup authored by the caller */
  .art-bd :global(.doc-md) {
    font-size: var(--fs-md);
    line-height: 1.62;
    color: var(--tx1);
    max-width: 72ch;
  }
  .art-bd :global(.doc-md h2) {
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
    color: var(--tx0);
    margin: 20px 0 7px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--bd0);
  }
  .art-bd :global(.doc-md > :first-child) {
    margin-top: 0;
  }
  .art-bd :global(.doc-md p) {
    margin: 7px 0;
  }
  .art-bd :global(.doc-md code) {
    font: 500 var(--fs-sm) var(--font-mono);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    padding: 0 4px;
    border-radius: 4px;
    color: var(--tx0);
  }
  .art-bd :global(.doc-md strong) {
    color: var(--tx0);
    font-weight: 600;
  }
  .art-bd :global(.ann-hl) {
    background: color-mix(in oklab, var(--st-warn) 22%, transparent);
    border-bottom: 2px solid var(--st-warn);
    border-radius: 2px;
    padding: 0 1px;
    cursor: pointer;
  }
  .art-bd :global(pre) {
    margin: 0;
    padding: 12px 16px;
    font: 400 var(--fs-sm) / 1.65 var(--font-mono);
    color: var(--tx1);
    overflow: auto;
    tab-size: 2;
  }

  .ic {
    width: 13px;
    height: 13px;
    flex: none;
    stroke: currentColor;
    fill: none;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }

  /* .t-artlaunch is the component-specific in-chat launch card; its Review
     button keeps an inline .t-btn (rendered alongside the card's own
     layout rather than pulled out to the shared Button). */
  .t-artlaunch {
    display: flex;
    align-items: center;
    gap: 11px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    background: var(--bg1);
    padding: 10px 12px;
    max-width: 520px;
  }
  .t-artlaunch .ai {
    width: 34px;
    height: 34px;
    flex: none;
    border-radius: var(--r-md);
    display: grid;
    place-items: center;
    background: color-mix(in oklab, var(--acc) 14%, var(--bg2));
    color: var(--acc);
  }
  .t-artlaunch .am {
    min-width: 0;
    flex: 1;
    display: grid;
    gap: 2px;
  }
  .t-artlaunch .am .t {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .t-artlaunch .am .s {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    height: 28px;
    padding: 0 11px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    background: var(--bg3);
    color: var(--tx0);
    font: 500 var(--fs-md) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
    transition:
      background var(--t-1) var(--ease),
      border-color var(--t-1) var(--ease),
      filter var(--t-1) var(--ease);
  }
  .t-btn--sm {
    height: 23px;
    padding: 0 8px;
    font-size: var(--fs-sm);
    gap: 5px;
  }
  .t-btn--sm .ic {
    width: 11px;
    height: 11px;
  }
  .t-btn--pri {
    background: var(--acc);
    border-color: transparent;
    color: var(--on-acc);
    font-weight: 600;
  }
  .t-btn--pri:hover {
    background: color-mix(in oklab, var(--acc) 88%, var(--tx0));
    border-color: transparent;
  }
</style>
