<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import { formatTokens, parseMarkdown } from './chatturns.util';
  import { rafThrottle } from './rafThrottle';

  let {
    label = 'thought',
    tokens,
    content = '',
    open = false,
  }: {
    // Summary shown on the toggle, e.g. "thought for 6s".
    label?: string;
    // Reasoning token count -> "· N tokens" meta.
    tokens?: number;
    // Markdown reasoning body.
    content?: string;
    open?: boolean;
  } = $props();

  // Follow the `open` prop as the preference flips; a manual toggle wins only
  // until the prop next changes.
  let userOverride = $state<{ prop: boolean; value: boolean } | null>(null);
  const isOpen = $derived(userOverride?.prop === open ? userOverride.value : open);

  // Seed synchronously (no flash of empty), then coalesce streamed reasoning
  // deltas to one parse per frame; the trailing value always renders.
  // svelte-ignore state_referenced_locally -- `content` seeds the first parse; the effect owns updates.
  let html = $state(parseMarkdown(content));
  const throttle = rafThrottle<string>((src) => (html = parseMarkdown(src)));

  let seeded = false;
  $effect(() => {
    const src = content;
    if (seeded) throttle.push(src);
    else seeded = true;
  });
  $effect(() => () => throttle.dispose());
</script>

<div class="t-think" class:is-open={isOpen}>
  <button
    type="button"
    aria-expanded={isOpen}
    onclick={() => (userOverride = { prop: open, value: !isOpen })}
  >
    <span class="chev"><Icon name="chev-r" size={10} /></span>
    {label}{#if tokens != null}<span class="tok"> · {formatTokens(tokens)} tokens</span>{/if}
  </button>
  <div class="body">
    <div>{@html html}</div>
  </div>
</div>

<style>
  .t-think {
    border-left: 2px solid color-mix(in oklab, var(--brand) 55%, transparent);
    padding: 2px 0 2px 10px;
    font-size: var(--fs-sm);
  }
  .t-think > button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: none;
    border: 0;
    padding: 2px 4px;
    margin-left: -4px;
    border-radius: var(--r-sm);
    color: var(--tx3);
    font: 500 var(--fs-xs) var(--font-mono);
    cursor: pointer;
  }
  .t-think > button:hover {
    color: var(--tx1);
    background: var(--bg2);
  }
  .t-think .tok {
    opacity: 0.85;
  }
  .t-think .chev {
    transition: rotate var(--t-2) var(--ease);
    display: inline-flex;
  }
  .t-think.is-open .chev {
    rotate: 90deg;
  }
  .t-think .body {
    display: grid;
    grid-template-rows: 0fr;
    transition: grid-template-rows var(--t-3) var(--ease);
  }
  .t-think.is-open .body {
    grid-template-rows: 1fr;
  }
  .t-think .body > div {
    overflow: hidden;
    color: var(--tx2);
    font-size: var(--fs-xs);
    line-height: 1.6;
    max-width: 64ch;
  }
  /* :global() - reasoning body is injected via {@html}. */
  .t-think .body :global(p) {
    margin: 6px 0 2px;
  }
  .t-think .body :global(code) {
    font-family: var(--font-mono);
    color: var(--tx0);
  }
  @media (prefers-reduced-motion: reduce) {
    .t-think .chev,
    .t-think .body {
      transition: none;
    }
  }
</style>
