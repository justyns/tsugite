<script lang="ts">
  // Activity feed: session runs, compactions, resolved jobs and schedule runs
  // merged by GET /api/activity. Filtering is a server round-trip, so a narrow
  // filter still fills the window.
  import { TESTID } from '$lib/testids';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import FeedRow from '$lib/components/rows/FeedRow.svelte';
  import { activity, type ActivityEntry } from '$lib/stores/activity.svelte';
  import { formatAgo } from '$lib/relativeTime';
  import { navigate } from '$lib/router.svelte';
  import { jobDrawerRequest } from '../jobs/jobDrawerSignal.svelte';
  import {
    ACTIVITY_FILTERS,
    type ActivityFilter,
    ENTRY_ICON,
    entryRoute,
    groupByDay,
  } from './activityFeed';

  let filter = $state<ActivityFilter>('all');
  let now = $state(Date.now());

  // Ticks the relative stamp without a refetch.
  $effect(() => {
    const t = setInterval(() => (now = Date.now()), 15_000);
    return () => clearInterval(t);
  });

  function refetch(): void {
    void activity.load(filter === 'all' ? {} : { types: filter });
  }

  $effect(() => {
    activity.rev; // track
    refetch();
  });

  const groups = $derived(groupByDay(activity.entries, now));
  const showSkeleton = $derived(activity.loading && activity.entries.length === 0);

  function open(entry: ActivityEntry): void {
    const route = entryRoute(entry);
    if (!route) return;
    // jobDrawerRequest is the same one-shot channel an in-chat job tile uses.
    if (entry.job_id) jobDrawerRequest.request(entry.job_id);
    navigate(route.view, route.params);
  }
</script>

<section class="activity-view" data-testid={TESTID.view('activity')} aria-labelledby="activity-h">
  <div class="act-tools">
    <h2 id="activity-h">Activity</h2>
    <div class="fpills" role="group" aria-label="Filter activity">
      {#each ACTIVITY_FILTERS as option (option.key)}
        <button
          type="button"
          class="fpill"
          class:is-active={filter === option.key}
          aria-pressed={filter === option.key}
          data-testid={TESTID.activityFilter(option.key)}
          onclick={() => (filter = option.key)}
        >
          {option.label}
        </button>
      {/each}
    </div>
  </div>

  <div class="act-body">
    {#if showSkeleton}
      <PaneState kind="loading" />
    {:else if activity.error && activity.entries.length === 0}
      <PaneState kind="error" title="Couldn't load activity">
        {#snippet icon()}<Icon name="alert" />{/snippet}
        {#snippet hint()}<span class="mono">{activity.error}</span>{/snippet}
        {#snippet actions()}
          <Button size="sm" data-testid={TESTID.activityRetry} onclick={refetch}>
            {#snippet icon()}<Icon name="retry" />{/snippet}Retry
          </Button>
        {/snippet}
      </PaneState>
    {:else if groups.length === 0}
      <PaneState kind="empty" title="No activity yet">
        {#snippet icon()}<Icon name="clock" />{/snippet}
        {#snippet hint()}Finished chats, resolved jobs, compactions and schedule runs land here.{/snippet}
      </PaneState>
    {:else}
      {#if activity.error}
        <div class="act-stale" role="status">
          <Icon name="alert" size={12} />
          <span class="mono">{activity.error}</span>
          <Button size="sm" data-testid={TESTID.activityRetry} onclick={refetch}>
            {#snippet icon()}<Icon name="retry" />{/snippet}Retry
          </Button>
        </div>
      {/if}
      <div class="feed" data-testid={TESTID.activityFeed}>
        {#each groups as group (group.day)}
          <section class="day" data-testid={TESTID.activityDay(group.day)}>
            <h3 class="day-h">{group.label}</h3>
            <ul class="rows">
              {#each group.entries as entry (entry.id)}
                <li>
                  <FeedRow
                    icon={ENTRY_ICON[entry.type]}
                    title={entry.title}
                    detail={entry.summary}
                    label={entry.label}
                    tone={entry.status}
                    when={formatAgo(entry.timestamp, now)}
                    testid={TESTID.activityEntry(entry.id)}
                    onopen={() => open(entry)}
                  />
                </li>
              {/each}
            </ul>
          </section>
        {/each}
      </div>
    {/if}
  </div>
</section>

<style>
  .activity-view {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .act-tools {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    flex-wrap: wrap;
    flex: none;
  }
  .act-tools h2 {
    margin: 0;
    font: 600 var(--fs-md) var(--font-ui);
    color: var(--tx0);
  }
  .fpills {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .fpill {
    display: inline-flex;
    align-items: center;
    height: 23px;
    padding: 0 9px;
    border-radius: var(--r-full);
    border: 1px solid var(--bd1);
    background: transparent;
    color: var(--tx2);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    cursor: pointer;
  }
  .fpill:hover {
    color: var(--tx0);
    border-color: var(--tx3);
  }
  .fpill:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .fpill.is-active {
    background: var(--bg3);
    color: var(--tx0);
    border-color: var(--bd1);
  }
  .act-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-4) var(--sp-5) 26px;
  }
  .mono {
    font-family: var(--font-mono);
  }
  .act-stale {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    margin-bottom: var(--sp-4);
    padding: var(--sp-2) var(--sp-3);
    border: 1px solid color-mix(in oklab, var(--st-warn) 45%, transparent);
    background: color-mix(in oklab, var(--st-warn) 8%, transparent);
    border-radius: var(--r-md);
    font-size: var(--fs-sm);
  }
  .act-stale .mono {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .feed {
    display: grid;
    gap: var(--sp-4);
    align-content: start;
  }
  .day-h {
    position: sticky;
    top: calc(var(--sp-4) * -1);
    z-index: 1;
    margin: 0 0 var(--sp-2);
    padding: 4px 0;
    background: var(--bg0);
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  .rows {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 3px;
  }
  @media (max-width: 640px) {
    .act-tools {
      gap: 6px;
      padding: 8px 10px;
    }
    .act-body {
      padding: var(--sp-3) var(--sp-3) 26px;
    }
  }
</style>
