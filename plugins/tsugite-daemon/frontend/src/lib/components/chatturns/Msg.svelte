<script lang="ts">
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';

  let {
    role,
    who,
    at,
    index,
    streaming = false,
    pinnedActs = false,
    retryFailed = false,
    onCopy,
    onEditFork,
    onRetry,
    children,
  }: {
    role: 'user' | 'ai';
    // Author label shown in the gutter, e.g. "you" / "tsugite".
    who: string;
    // Timestamp shown under the author, e.g. "14:22".
    at: string;
    // Optional turn index, rendered as "#NN" alongside the timestamp.
    index?: number;
    // Marks the body as a live region while tokens stream in.
    streaming?: boolean;
    // Keep the hover action bar visible (e.g. after a menu opens on touch).
    pinnedActs?: boolean;
    // This AI turn FAILED: surface Retry as a prominent, always-visible button
    // under the error instead of the hover-only regenerate icon (a dead-end turn
    // must not hide its recovery affordance behind a hover).
    retryFailed?: boolean;
    onCopy?: () => void;
    onEditFork?: () => void;
    onRetry?: () => void;
    children: Snippet;
  } = $props();

  let bodEl: HTMLElement | undefined;

  async function copy() {
    try {
      await navigator.clipboard?.writeText(bodEl?.textContent ?? '');
    } catch {
      // Clipboard unavailable - fall through to the callback.
    }
    onCopy?.();
  }
</script>

<article class="t-msg t-msg--{role}">
  <div class="gut">
    <span class="who">{who}</span>
    <span class="at">{at}</span>
    {#if index != null}<span class="idx">#{String(index).padStart(2, '0')}</span>{/if}
  </div>
  <div
    class="bod"
    bind:this={bodEl}
    aria-live={streaming ? 'polite' : null}
    aria-busy={streaming ? true : null}
  >
    {@render children()}
  </div>
  {#if role === 'ai' && onRetry && retryFailed}
    <!-- The turn failed: Retry is the recovery action, so it reads as a real
         labelled button under the error - never hover-gated like the regenerate
         icon a healthy turn carries. Re-sends the last user message (onRetry). -->
    <div class="retry-failed">
      <button type="button" class="retry-btn" data-testid={TESTID.chatRetry} onclick={onRetry}>
        <Icon name="retry" size={12} />Retry
      </button>
    </div>
  {/if}
  <div class="acts" class:is-pinned={pinnedActs}>
    <button type="button" class="t-iconbtn" aria-label="Copy message" onclick={copy}>
      <Icon name="copy" size={11} />
    </button>
    {#if role === 'user' && onEditFork}
      <button
        type="button"
        class="t-iconbtn"
        aria-label="Edit and fork from here"
        onclick={onEditFork}
      >
        <Icon name="edit" size={11} />edit &amp; fork
      </button>
    {:else if role === 'ai' && onRetry && !retryFailed}
      <button type="button" class="t-iconbtn" aria-label="Retry this response" onclick={onRetry}>
        <Icon name="retry" size={11} />
      </button>
    {/if}
  </div>
</article>

<style>
  .t-msg {
    display: grid;
    grid-template-columns: 76px minmax(0, 1fr);
    gap: 4px 14px;
    padding: 12px 18px 12px 14px;
    position: relative;
    border-top: 1px solid color-mix(in oklab, var(--bd0) 55%, transparent);
  }
  .t-msg--user {
    background: color-mix(in oklab, var(--acc) 5%, transparent);
  }
  .t-msg .gut {
    display: flex;
    flex-direction: column;
    gap: 2px;
    align-items: flex-end;
    text-align: right;
  }
  .t-msg .who {
    font: 600 var(--fs-2xs) / 1.6 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .t-msg--user .who {
    color: var(--acc);
  }
  .t-msg--ai .who {
    color: var(--brand);
  }
  .t-msg .at {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    opacity: 0.8;
  }
  .t-msg .idx {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    opacity: 0.6;
  }
  .t-msg .bod {
    display: grid;
    /* An explicit minmax(0,1fr) column instead of the implicit `auto` one: `auto`
       sizes to the widest block's max-content, so a wide code block or long token
       would stretch the whole turn past a narrow pane. minmax(0,1fr) pins the
       column to the container and lets wide blocks scroll inside themselves. */
    grid-template-columns: minmax(0, 1fr);
    gap: 9px;
    min-width: 0;
  }
  .t-msg .acts {
    position: absolute;
    top: 8px;
    right: 12px;
    display: none;
    gap: 2px;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 2px;
    box-shadow: var(--sh-1);
  }
  .t-msg:hover .acts,
  .t-msg:focus-within .acts,
  .t-msg .acts.is-pinned {
    display: inline-flex;
  }

  /* Prominent Retry on a failed turn: sits under the body (grid column 2), so it
     reads as the recovery action right below the error block. Always visible. */
  .retry-failed {
    grid-column: 2;
    display: flex;
  }
  .retry-btn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: color-mix(in oklab, var(--acc) 12%, transparent);
    border: 1px solid color-mix(in oklab, var(--acc) 40%, transparent);
    color: var(--acc);
    font: 600 var(--fs-xs) var(--font-mono);
    cursor: pointer;
    padding: 5px 12px;
    border-radius: var(--r-md);
  }
  .retry-btn:hover {
    background: color-mix(in oklab, var(--acc) 20%, transparent);
    border-color: var(--acc);
  }

  /* Narrow: the gutter stacks above the body as a row. */
  @media (max-width: 640px) {
    .t-msg {
      grid-template-columns: 1fr;
      gap: 2px;
      padding: 10px 12px;
    }
    .t-msg .gut {
      flex-direction: row;
      align-items: baseline;
      gap: 8px;
    }
    /* Single-column grid at this width: the failed-turn Retry drops back to col 1. */
    .retry-failed {
      grid-column: 1;
    }
  }

  /* .t-iconbtn is a bare icon-button with no shared component (Toast keeps its
     own too); its 11px icon sizing lives globally in tokens.css (.t-iconbtn .ic). */
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
</style>
