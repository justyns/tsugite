<script lang="ts">
  // Usage & cost dashboard: a dense table of days × tokens/cost with bar meters
  // (role=meter); range picker as segmented control; totals row pinned. No
  // cost-trend bar chart - the per-day Meter bars carry that job.
  //
  // No aria-label/aria-labelledby on the root section: the mux Pane already
  // landmarks this region via its tab title, so a second same-named region would
  // just duplicate the landmark (see views/tools/View.svelte).
  import { untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import { usage, type UsageCacheSplit } from '$lib/stores/usage.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Meter from '$lib/components/datadisplay/Meter.svelte';
  import Table, { type TableColumn } from '$lib/components/datadisplay/Table.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import ProviderQuota from './ProviderQuota.svelte';
  import {
    formatDayLabel,
    formatLastRun,
    formatRuns,
    formatTokensCompact,
    formatUsd,
  } from './format';

  const RANGE_OPTIONS = ['7 days', '30 days', '90 days'] as const;
  const RANGE_DAYS: Record<(typeof RANGE_OPTIONS)[number], number> = {
    '7 days': 7,
    '30 days': 30,
    '90 days': 90,
  };
  let rangeLabel = $state<(typeof RANGE_OPTIONS)[number]>('30 days');

  // Fires the initial load and every range switch. `usage.load()` reads its own
  // `range` field (to merge the partial patch) before writing it back, so calling
  // it un-untracked would make this effect depend on `usage.range` too - and since
  // the same call *writes* that field a moment later, it would immediately
  // re-trigger itself. `untrack` keeps the mutation out of the effect's dependency
  // set, matching the pattern already used for App.svelte's deep-link effect.
  $effect(() => {
    const days = RANGE_DAYS[rangeLabel];
    untrack(() => usage.load({ sinceDays: days }));
  });

  const hasData = $derived(
    usage.summary.length > 0 ||
      usage.agents.length > 0 ||
      usage.models.length > 0 ||
      usage.schedules.length > 0,
  );
  const maxTokens = $derived(Math.max(1, ...usage.summary.map((r) => r.total_tokens ?? 0)));
  const maxCost = $derived(Math.max(1, ...usage.summary.map((r) => r.total_cost ?? 0)));

  const CACHE_COLS: TableColumn[] = [
    { key: 'cache_read', label: 'cache rd', align: 'right' },
    { key: 'cache_write', label: 'cache wr', align: 'right' },
  ];
  const COST_COL: TableColumn = { key: 'cost', label: 'cost', align: 'right' };
  const agentColumns: TableColumn[] = [
    { key: 'agent', label: 'agent' },
    { key: 'runs', label: 'runs', align: 'right' },
    ...CACHE_COLS,
    COST_COL,
  ];
  const modelColumns: TableColumn[] = [
    { key: 'model', label: 'model' },
    { key: 'tokens', label: 'tokens', align: 'right' },
    ...CACHE_COLS,
    COST_COL,
  ];
  const scheduleColumns: TableColumn[] = [
    { key: 'schedule', label: 'schedule' },
    { key: 'runs', label: 'runs', align: 'right' },
    { key: 'tokens', label: 'tokens', align: 'right' },
    ...CACHE_COLS,
    COST_COL,
    { key: 'last_run', label: 'last run', align: 'right' },
  ];
  const dayColumns: TableColumn[] = [
    { key: 'day', label: 'day' },
    { key: 'runs', label: 'runs', align: 'right' },
    { key: 'tokens', label: 'tokens' },
    ...CACHE_COLS,
    { key: 'cost', label: 'cost' },
  ];

  function retry() {
    usage.load();
  }
</script>

<section
  data-testid={TESTID.view('usage')}
  class="usage-view"
  aria-busy={usage.loading || undefined}
>
  <div class="u-head">
    <div class="row">
      <h2>Usage &amp; cost</h2>
      <Seg options={[...RANGE_OPTIONS]} bind:value={rangeLabel} ariaLabel="Range" />
      <div class="grow"></div>
      {#if usage.total}
        <p class="u-totals mono">
          runs <b>{formatRuns(usage.total.runs)}</b> · tokens
          <b>{formatTokensCompact(usage.total.total_tokens)}</b> · cost
          <b>{formatUsd(usage.total.total_cost)}</b>
        </p>
      {/if}
    </div>
  </div>

  {#if usage.providers.length > 0}
    <div>
      <h4 class="d-sec-h">provider limits</h4>
      <div class="u-quota">
        {#each usage.providers as row (row.provider)}
          <div class="u-prov"><ProviderQuota {row} /></div>
        {/each}
      </div>
    </div>
  {/if}

  {#if usage.loading && !hasData}
    <PaneState kind="loading" lines={6} />
  {:else if usage.error}
    <PaneState kind="error" title="Couldn't load usage">
      {#snippet icon()}<Icon name="alert" />{/snippet}
      {#snippet hint()}<span class="mono">{usage.error}</span>{/snippet}
      {#snippet actions()}
        <Button size="sm" onclick={retry}>
          {#snippet icon()}<Icon name="retry" />{/snippet}
          Retry
        </Button>
      {/snippet}
    </PaneState>
  {:else if !hasData}
    <PaneState kind="empty" title="No usage recorded yet">
      {#snippet icon()}<Icon name="usage" />{/snippet}
      {#snippet hint()}<span>Nothing logged in the last {RANGE_DAYS[rangeLabel]} days.</span
        >{/snippet}
    </PaneState>
  {:else}
    {#snippet cacheCells(row: UsageCacheSplit)}
      <td class="tr mono c2" title="cache reads (cheap cache hits)">
        {formatTokensCompact(row.cache_read_tokens)}
      </td>
      <td class="tr mono c2" title="cache creation (writes)">
        {formatTokensCompact(row.cache_creation_tokens)}
      </td>
    {/snippet}

    <div class="u-cols">
      <div>
        <h4 class="d-sec-h">top agents</h4>
        <div class="tbl-wrap">
          <Table
            columns={agentColumns}
            rows={usage.agents}
            key={(row) => row.agent}
            ariaLabel="Top agents by cost"
          >
            {#snippet cells(row)}
              <td>{row.agent}</td>
              <td class="tr mono c2">{formatRuns(row.runs)}</td>
              {@render cacheCells(row)}
              <td class="tr mono">{formatUsd(row.total_cost)}</td>
            {/snippet}
          </Table>
        </div>
      </div>
      <div>
        <h4 class="d-sec-h">top models</h4>
        <div class="tbl-wrap">
          <Table
            columns={modelColumns}
            rows={usage.models}
            key={(row) => row.model}
            ariaLabel="Top models by cost"
          >
            {#snippet cells(row)}
              <td class="mono">{row.model}</td>
              <td class="tr mono c2">{formatTokensCompact(row.total_tokens)}</td>
              {@render cacheCells(row)}
              <td class="tr mono">{formatUsd(row.total_cost)}</td>
            {/snippet}
          </Table>
        </div>
      </div>
    </div>

    {#if usage.schedules.length > 0}
      <div>
        <h4 class="d-sec-h">scheduled tasks</h4>
        <div class="tbl-wrap">
          <Table
            columns={scheduleColumns}
            rows={usage.schedules}
            key={(row) => row.schedule_name ?? 'unattributed'}
            ariaLabel="Usage by scheduled task"
          >
            {#snippet cells(row)}
              <td class="mono">{row.schedule_name ?? '(unattributed)'}</td>
              <td class="tr mono c2">{formatRuns(row.runs)}</td>
              <td class="tr mono c2">{formatTokensCompact(row.total_tokens)}</td>
              {@render cacheCells(row)}
              <td class="tr mono">{formatUsd(row.total_cost)}</td>
              <td class="tr mono c2">{formatLastRun(row.last_run)}</td>
            {/snippet}
          </Table>
        </div>
      </div>
    {/if}

    <div>
      <h4 class="d-sec-h">detail · per day</h4>
      <div class="tbl-wrap u-day-wrap">
        <Table
          columns={dayColumns}
          rows={usage.summary}
          key={(row) => row.period}
          ariaLabel="Usage per day"
        >
          {#snippet cells(row)}
            {@const dayLabel = formatDayLabel(row.period)}
            <td class="mono">{dayLabel}</td>
            <td class="tr mono c2">{formatRuns(row.runs)}</td>
            <td>
              <Meter
                value={row.total_tokens ?? 0}
                max={maxTokens}
                label="{dayLabel} tokens"
                displayText={formatTokensCompact(row.total_tokens)}
              />
            </td>
            {@render cacheCells(row)}
            <td>
              <Meter
                value={row.total_cost ?? 0}
                max={maxCost}
                label="{dayLabel} cost"
                displayText={formatUsd(row.total_cost)}
              />
            </td>
          {/snippet}
        </Table>
      </div>
    </div>
  {/if}
</section>

<style>
  .usage-view {
    display: grid;
    gap: var(--sp-4);
    padding: 14px 16px 26px;
    align-content: start;
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-3);
    flex-wrap: wrap;
  }
  .u-head {
    /* Pinned totals: sticks to the top of the pane's scrollport as the day
       table scrolls beneath it. The negative margin cancels this grid's own
       padding so the bar sits flush against the scrollport edge once stuck. */
    position: sticky;
    top: -14px;
    z-index: 3;
    margin: -14px -16px 0;
    padding: 14px 16px var(--sp-3);
    background: var(--bg0);
    border-bottom: 1px solid var(--bd0);
  }
  .u-head h2 {
    margin: 0;
    font: 600 var(--fs-lg) var(--font-ui);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .u-totals {
    margin: 0;
    font-size: var(--fs-xs);
    color: var(--tx3);
    white-space: nowrap;
  }
  .u-totals b {
    color: var(--tx1);
    font-weight: 600;
  }

  .u-cols {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--sp-4);
  }

  /* Provider quota cards, on the .t-job recipe. */
  .u-quota {
    display: grid;
    /* auto-fill, not auto-fit: one provider keeps a card's width instead of
       stretching across the pane. */
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
    gap: var(--sp-3);
  }
  .u-prov {
    background: var(--bg2);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 9px 10px;
    min-width: 0;
  }

  .d-sec-h {
    margin: 0 0 7px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }

  .tbl-wrap {
    overflow-x: auto;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
  }
  .u-day-wrap {
    max-height: 480px;
    overflow-y: auto;
  }
  .mono {
    font-family: var(--font-mono);
  }
</style>
