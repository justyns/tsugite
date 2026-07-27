<script lang="ts">
  // List layout. The shared `Table` primitive only takes string / static-snippet
  // cells and has no per-row testid or row-open affordance, so the data-driven
  // job table is built here from the `.t-table` contract. The row's
  // actionable element is a real button in the job cell (a11y-correct: the row
  // itself stays informational).
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import type { Job } from '$lib/stores/jobs.svelte';
  import JobPill from './JobPill.svelte';
  import JobCrit from './JobCrit.svelte';
  import { acCounts, acRows, attemptCount } from './jobModel';
  import { relativeTime } from './format';
  import { boardColForState, type SortMode } from './board';

  let {
    jobs,
    now,
    selectedId,
    sortMode,
    sortDir,
    onOpen,
    onSortUpdated,
  }: {
    jobs: Job[];
    now: number;
    selectedId: string | null;
    sortMode: SortMode;
    sortDir: 'ascending' | 'descending';
    onOpen: (job: Job) => void;
    onSortUpdated: () => void;
  } = $props();

  const updatedSort = $derived(sortMode === 'updated' ? sortDir : 'none');
  function attemptText(job: Job): string {
    const n = attemptCount(job);
    if (n === 0) return '—';
    return `${n}/${job.max_attempts}`;
  }
</script>

<table class="t-table" data-testid={TESTID.jobsTable} aria-label="Jobs list">
  <thead>
    <tr>
      <th scope="col">job</th>
      <th scope="col">state</th>
      <th scope="col">attempt</th>
      <th scope="col">criteria</th>
      <th scope="col">agent</th>
      <th scope="col" class="sortable" aria-sort={updatedSort}>
        <button type="button" onclick={onSortUpdated}>updated<Icon name="chev-d" size={9} /></button
        >
      </th>
    </tr>
  </thead>
  <tbody>
    {#each jobs as job (job.job_id)}
      {@const counts = acCounts(acRows(job))}
      <tr
        class:is-selected={job.job_id === selectedId}
        class:is-off={boardColForState(job.state) === 'resolved'}
      >
        <td class="jl-job">
          <button
            type="button"
            class="jl-open"
            data-testid={TESTID.jobRow(job.job_id)}
            aria-pressed={job.job_id === selectedId}
            onclick={() => onOpen(job)}
          >
            {job.prompt}
          </button>
        </td>
        <td><JobPill st={job.state} /></td>
        <td class="c2 mono">{attemptText(job)}</td>
        <td>
          {#if counts.total > 0}<JobCrit {counts} />{:else}<span class="c3 mono">—</span>{/if}
        </td>
        <td class="c2">{job.agent}</td>
        <td class="c3 mono">{relativeTime(job.updated_at, now) || '—'}</td>
      </tr>
    {/each}
  </tbody>
</table>

<style>
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
  .t-table th :global(.ic) {
    vertical-align: -1px;
  }
  .t-table td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--bd0);
    height: 34px;
    vertical-align: middle;
    white-space: nowrap;
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
  /* The job cell's open button reads as plain text but is keyboard-focusable. */
  .jl-open {
    all: unset;
    cursor: pointer;
    color: var(--tx0);
    max-width: 42ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: block;
  }
  .jl-open:hover {
    color: var(--acc);
  }
  .jl-open:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
    border-radius: var(--r-sm);
  }
  .jl-job {
    max-width: 42ch;
  }
</style>
