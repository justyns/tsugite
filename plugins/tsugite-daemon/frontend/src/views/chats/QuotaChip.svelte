<script lang="ts">
  // Header quota chip: the most-used subscription window of the chat's provider,
  // opening a popover with every window. Renders nothing when that provider
  // reports no quota.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { clampIntoView } from '$lib/actions/clampIntoView';
  import { buildHash } from '$lib/router.svelte';
  import { usage } from '$lib/stores/usage.svelte';
  import { TESTID } from '$lib/testids';
  import ProviderQuota from '../usage/ProviderQuota.svelte';
  import { QUOTA_WARN_PCT, formatPercent } from '../usage/format';

  let {
    provider,
    working = false,
  }: {
    provider: string | null;
    working?: boolean;
  } = $props();

  let root = $state<HTMLElement>();
  let open = $state(false);

  $effect(() => {
    if (provider && !working) usage.refreshProviders();
  });

  const report = $derived(usage.providers.find((r) => r.provider === provider) ?? null);
  const top = $derived(report?.windows.toSorted((a, b) => b.used_pct - a.used_pct)[0]);
  const warn = $derived((top?.used_pct ?? 0) >= QUOTA_WARN_PCT);
  const text = $derived(top ? `${top.label} ${formatPercent(top.used_pct)}` : 'quota');

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

{#if report && (report.error || top)}
  <div class="quota-anchor" bind:this={root}>
    <button
      type="button"
      class="q-chip"
      class:is-warn={warn}
      class:is-err={!!report.error}
      aria-haspopup="dialog"
      aria-expanded={open}
      aria-label="{report.label} quota {report.error ? 'unavailable' : text}"
      title={report.error ?? undefined}
      data-testid={TESTID.chatQuota}
      onclick={() => (open = !open)}
    >
      {#if warn || report.error}<Icon name="alert" size={11} />{/if}{text}
    </button>
    {#if open}
      <div
        class="q-pop"
        role="dialog"
        aria-label="{report.label} quota"
        data-testid={TESTID.chatQuotaPopover}
        use:clampIntoView
      >
        <div class="q-body"><ProviderQuota row={report} /></div>
        <a class="q-link" href={buildHash('usage')} onclick={() => (open = false)}>open usage</a>
      </div>
    {/if}
  </div>
{/if}

<style>
  .quota-anchor {
    position: relative;
    display: inline-flex;
    flex: none;
  }
  .q-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    cursor: pointer;
  }
  .q-chip:hover {
    color: var(--acc);
  }
  .q-chip.is-warn {
    color: var(--st-warn);
  }
  .q-chip.is-err {
    color: var(--tx3);
  }
  .q-pop {
    position: absolute;
    top: calc(100% + 6px);
    right: 0;
    z-index: 60;
    width: min(360px, 82vw);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    overflow: hidden;
  }
  .q-body {
    padding: 9px 11px;
  }
  .q-link {
    display: block;
    padding: 7px 11px;
    border-top: 1px solid var(--bd0);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx2);
  }
  .q-link:hover {
    color: var(--tx0);
  }
</style>
