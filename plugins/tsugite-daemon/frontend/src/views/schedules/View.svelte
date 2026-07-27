<script lang="ts">
  // Schedules view: a sortable table of cron / one-off tasks with inline
  // enable-disable + run-now, a create/edit drawer (agent vs script, cron vs
  // once), and a cleanup action for auto-disabled entries. The table markup is
  // its own `.t-table` rather than the shared <Table>:
  // its rows are click-to-open and carry rich interactive cells (switch, pill,
  // sparkline, run button) that <Table>'s zero-arg string/snippet cell model and
  // non-interactive rows can't express.
  import { TESTID } from '$lib/testids';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Spark from '$lib/components/datadisplay/Spark.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { schedules, type Schedule } from '$lib/stores/schedules.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { describeCron } from './cron';
  import {
    deriveRunStatus,
    sortSchedules,
    summarize,
    nextUp,
    buildSpark,
    formatNextRun,
    formatAgo,
    formatStamp,
    type SortDir,
  } from './schedulesView';
  import RunStatusPill from './RunStatusPill.svelte';
  import EnableSwitch from './EnableSwitch.svelte';
  import ScheduleDrawer from './ScheduleDrawer.svelte';

  // Live clock so relative countdowns tick without a full reload. Minute-ish
  // resolution is enough for "in 4m" / "3d ago".
  let now = $state(Date.now());
  $effect(() => {
    const t = setInterval(() => (now = Date.now()), 15_000);
    return () => clearInterval(t);
  });

  // Initial fetch (no reactive deps -> runs once on mount). Live schedule_update
  // broadcasts already refresh the store list via the shell event router.
  $effect(() => {
    void schedules.load();
    void agentsMeta.load();
  });

  const agentNames = $derived(agentsMeta.agents.map((a) => a.name));

  let sortDir = $state<SortDir>('ascending');
  const sorted = $derived(sortSchedules(schedules.list, sortDir));
  const summary = $derived(summarize(schedules.list));
  const upcoming = $derived(nextUp(schedules.list, now));

  function flipSort() {
    sortDir = sortDir === 'ascending' ? 'descending' : 'ascending';
  }

  const summaryText = $derived(
    [
      `${summary.total} scheduled`,
      summary.failing ? `${summary.failing} failing` : '',
      summary.disabled ? `${summary.disabled} disabled` : '',
    ]
      .filter(Boolean)
      .join(' · '),
  );

  function cadence(s: Schedule): { expr: string; human: string | null } {
    if (s.schedule_type === 'once') {
      return { expr: 'once', human: s.run_at ? formatStamp(s.run_at) : 'no run time' };
    }
    return { expr: s.cron_expr ?? '', human: describeCron(s.cron_expr) };
  }

  // --- drawer (create / edit) ---
  let drawerOpen = $state(false);
  let selectedId = $state<string | null>(null); // null => create mode
  const selected = $derived(
    selectedId ? (schedules.list.find((s) => s.id === selectedId) ?? null) : null,
  );

  function openCreate() {
    selectedId = null;
    drawerOpen = true;
  }
  function openEdit(s: Schedule) {
    selectedId = s.id;
    drawerOpen = true;
  }
  function closeDrawer() {
    drawerOpen = false;
    selectedId = null;
  }

  // --- row actions ---
  async function toggle(s: Schedule, next: boolean) {
    try {
      if (next) await schedules.enable(s.id);
      else await schedules.disable(s.id);
      toasts.push('info', next ? `Enabled: ${s.id}` : `Disabled: ${s.id}`, {
        body: next ? 'next run recalculated' : "won't run until re-enabled",
      });
      await schedules.load();
    } catch (err) {
      toasts.push('err', 'Could not update schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function runNow(s: Schedule) {
    try {
      await schedules.run(s.id);
      toasts.push('info', 'Run queued', { body: `${s.id} starts on the next tick` });
      await schedules.load();
    } catch (err) {
      toasts.push('err', 'Could not run schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // --- cleanup ---
  let cleaning = $state(false);
  async function cleanup() {
    if (cleaning) return;
    cleaning = true;
    try {
      const removed = await schedules.cleanup();
      if (removed.length === 0) {
        toasts.push('info', 'Nothing to clean up', { body: 'no auto-disabled schedules' });
      } else {
        toasts.push('ok', `Removed ${removed.length} schedule${removed.length === 1 ? '' : 's'}`, {
          body: removed.join(', '),
        });
      }
      await schedules.load();
    } catch (err) {
      toasts.push('err', 'Cleanup failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      cleaning = false;
    }
  }

  // --- delete (confirmed) ---
  let pendingDelete = $state<Schedule | null>(null);
  async function confirmDelete() {
    const s = pendingDelete;
    pendingDelete = null;
    if (!s) return;
    try {
      await schedules.remove(s.id);
      if (selectedId === s.id) closeDrawer();
      toasts.push('info', 'Schedule deleted', { body: s.id });
      await schedules.load();
    } catch (err) {
      toasts.push('err', 'Could not delete schedule', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  const hasError = $derived(schedules.error != null && schedules.list.length === 0);
  const isLoading = $derived(schedules.loading && schedules.list.length === 0);
  const isEmpty = $derived(!schedules.loading && !schedules.error && schedules.list.length === 0);
</script>

<section class="sched" data-testid={TESTID.view('schedules')}>
  <div class="sched-hd">
    <strong>Schedules</strong>
    <span class="dim mono">{summaryText}</span>
    <div class="grow"></div>
    <Button
      size="sm"
      variant="ghost"
      loading={cleaning}
      data-testid={TESTID.schedulesCleanup}
      title="Remove auto-disabled schedules (expired / max-runs / missed)"
      onclick={cleanup}
    >
      Clean up
    </Button>
    <Button size="sm" data-testid={TESTID.schedulesNew} onclick={openCreate}>
      {#snippet icon()}<Icon name="plus" />{/snippet}
      New schedule
    </Button>
  </div>

  <div class="sched-wrap">
    {#if isLoading}
      <PaneState kind="loading" lines={5} />
    {:else if hasError}
      <PaneState kind="error" title="Couldn't load schedules">
        {#snippet hint()}<span>{schedules.error}</span>{/snippet}
        {#snippet actions()}
          <Button size="sm" onclick={() => schedules.load()}>Retry</Button>
        {/snippet}
      </PaneState>
    {:else if isEmpty}
      <PaneState kind="empty" title="No schedules yet">
        {#snippet icon()}<Icon name="sched" />{/snippet}
        {#snippet hint()}
          <span>Schedule an agent or a shell command to run on a cron cadence or once.</span>
        {/snippet}
        {#snippet actions()}
          <Button size="sm" variant="pri" onclick={openCreate}>
            {#snippet icon()}<Icon name="plus" />{/snippet}
            New schedule
          </Button>
        {/snippet}
      </PaneState>
    {:else}
      <table class="t-table" aria-label="Schedules" data-testid={TESTID.schedulesTable}>
        <thead>
          <tr>
            <th scope="col"><span class="vh">enabled</span></th>
            <th scope="col">schedule</th>
            <th scope="col">cadence</th>
            <th scope="col">last run</th>
            <th scope="col">history</th>
            <th scope="col" class="sortable" aria-sort={sortDir}>
              <button type="button" data-testid={TESTID.schedulesSortNext} onclick={flipSort}>
                next run<Icon name="chev-d" size={9} />
              </button>
            </th>
            <th scope="col"><span class="vh">actions</span></th>
          </tr>
        </thead>
        <tbody>
          {#each sorted as s (s.id)}
            {@const cad = cadence(s)}
            {@const spark = buildSpark(s.run_history)}
            <!-- svelte-ignore a11y_click_events_have_key_events -->
            <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
            <tr
              class:is-off={!s.enabled}
              class:is-selected={selectedId === s.id}
              data-testid={TESTID.scheduleRow(s.id)}
              onclick={() => openEdit(s)}
            >
              <td>
                <EnableSwitch
                  checked={s.enabled}
                  ariaLabel={`${s.id} enabled`}
                  testid={TESTID.scheduleToggle(s.id)}
                  onToggle={(next) => toggle(s, next)}
                />
              </td>
              <td>
                <button
                  type="button"
                  class="namebtn"
                  onclick={(e) => {
                    e.stopPropagation();
                    openEdit(s);
                  }}
                >
                  {s.id}
                </button>
                <span class="desc c3">{s.prompt}</span>
              </td>
              <td class="c2 mono">
                {cad.expr}
                {#if cad.human}<span class="cronh">{cad.human}</span>{/if}
              </td>
              <td>
                <span class="lastrun">
                  <RunStatusPill status={deriveRunStatus(s)} />
                  <span class="ago c3 mono">{formatAgo(s.last_run, now)}</span>
                </span>
              </td>
              <td><Spark points={spark.points} label={spark.label} /></td>
              <td class="c2 mono"
                >{formatNextRun(s.enabled ? s.next_run : null, s.timezone, now)}</td
              >
              <td>
                <Button
                  size="sm"
                  variant="ghost"
                  iconOnly
                  aria-label={`Run ${s.id} now`}
                  data-testid={TESTID.scheduleRunNow(s.id)}
                  onclick={(e) => {
                    e.stopPropagation();
                    runNow(s);
                  }}
                >
                  {#snippet icon()}<Icon name="play" />{/snippet}
                </Button>
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    {/if}
  </div>

  {#if !isEmpty && !isLoading && !hasError}
    <div class="sched-ft">
      {#if upcoming}
        <span
          >next: <b class="mono">{upcoming.schedule.id}</b>
          {formatNextRun(upcoming.schedule.next_run, upcoming.schedule.timezone, now)}</span
        >
      {:else}
        <span>no upcoming runs</span>
      {/if}
      <span>runner: local daemon</span>
      <div class="grow"></div>
      <span>sorted by next run — click header to flip</span>
    </div>
  {/if}

  <ScheduleDrawer
    open={drawerOpen}
    schedule={selected}
    agents={agentNames}
    onclose={closeDrawer}
    onchanged={() => schedules.load()}
    onRequestDelete={(s) => (pendingDelete = s)}
  />
</section>

<Modal
  open={pendingDelete != null}
  onclose={() => (pendingDelete = null)}
  title="Delete schedule?"
  tone="danger"
>
  {#snippet children()}
    <p>
      Delete <code>{pendingDelete?.id}</code>? It won't run again. Run history is retained by the
      daemon.
    </p>
  {/snippet}
  {#snippet footer()}
    <Button size="sm" data-autofocus onclick={() => (pendingDelete = null)}>Cancel</Button>
    <Button size="sm" variant="danger" onclick={confirmDelete}>Delete schedule</Button>
  {/snippet}
</Modal>

<style>
  .sched {
    position: relative;
    display: flex;
    flex-direction: column;
    height: 100%;
    min-height: 0;
    overflow: hidden;
    background: var(--bg1);
  }
  .sched-hd {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    flex: none;
  }
  .sched-hd strong {
    font: 600 var(--fs-sm) var(--font-ui);
  }
  .dim {
    color: var(--tx3);
    font-size: var(--fs-2xs);
  }
  .grow {
    flex: 1;
  }
  .sched-wrap {
    flex: 1;
    min-height: 0;
    overflow: auto;
    padding: 0;
  }
  /* PaneState sits in the scroll body with breathing room. */
  .sched-wrap > :global(.t-pane),
  .sched-wrap > :global(.pane-loading) {
    margin: 14px;
  }
  .sched-ft {
    display: flex;
    gap: 14px;
    align-items: center;
    padding: 6px 12px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    flex: none;
  }
  .sched-ft b {
    color: var(--tx1);
  }

  /* ---- table (.t-table) ---- */
  .t-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-sm);
  }
  .t-table th {
    position: sticky;
    top: 0;
    z-index: 2;
    background: var(--bg1);
    text-align: left;
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
    padding: 7px 10px;
    border-bottom: 1px solid var(--bd1);
    white-space: nowrap;
  }
  .t-table th.sortable button {
    all: unset;
    display: inline-flex;
    align-items: center;
    gap: 2px;
    cursor: pointer;
    font: inherit;
    color: inherit;
    text-transform: inherit;
    letter-spacing: inherit;
  }
  .t-table th.sortable button:hover {
    color: var(--tx1);
  }
  .t-table th.sortable button:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
    border-radius: var(--r-sm);
  }
  .t-table td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--bd0);
    height: 34px;
    vertical-align: middle;
    white-space: nowrap;
  }
  .t-table tbody tr {
    cursor: pointer;
  }
  .t-table tbody tr:hover {
    background: color-mix(in oklab, var(--bg3) 45%, transparent);
  }
  .t-table tbody tr.is-selected {
    background: color-mix(in oklab, var(--acc) 11%, transparent);
    box-shadow: inset 2px 0 0 0 var(--acc);
  }
  .t-table tbody tr.is-off {
    opacity: 0.55;
  }
  .t-table .c2 {
    color: var(--tx2);
  }
  .t-table .c3 {
    color: var(--tx3);
  }
  .t-table .mono {
    font-family: var(--font-mono);
  }

  /* schedule (name + description) cell */
  .namebtn {
    all: unset;
    font-weight: 600;
    color: var(--tx0);
    cursor: pointer;
    display: inline-block;
  }
  .namebtn:hover {
    color: var(--acc);
  }
  .namebtn:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
    border-radius: var(--r-sm);
  }
  .desc {
    display: block;
    font-size: var(--fs-xs);
    max-width: 32ch;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .cronh {
    display: block;
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
  }
  .lastrun {
    display: inline-flex;
    align-items: center;
    gap: 7px;
  }
  .ago {
    font-size: var(--fs-2xs);
  }

  /* visually-hidden header text for the icon-only columns */
  .vh {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }

  /* Narrow: shed the cadence + history columns instead of panning. */
  @media (max-width: 640px) {
    .sched-wrap :global(:is(td, th):nth-child(3)),
    .sched-wrap :global(:is(td, th):nth-child(5)) {
      display: none;
    }
    .sched-wrap :global(td) {
      white-space: normal;
      padding: 5px 6px;
    }
  }
</style>
