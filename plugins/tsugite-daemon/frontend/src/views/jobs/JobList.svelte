<script lang="ts">
  import Table, { type TableColumn } from '$lib/components/datadisplay/Table.svelte';
  import { TESTID } from '$lib/testids';
  import type { Job } from '$lib/stores/jobs.svelte';
  import JobPill from './JobPill.svelte';
  import JobCrit from './JobCrit.svelte';
  import { acCounts, acRows, attemptCount } from './jobModel';
  import { formatAgo } from '$lib/relativeTime';
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

  const columns: TableColumn[] = [
    { key: 'job', label: 'job' },
    { key: 'state', label: 'state' },
    { key: 'attempt', label: 'attempt' },
    { key: 'criteria', label: 'criteria' },
    { key: 'agent', label: 'agent' },
    { key: 'updated', label: 'updated', sortable: true },
  ];
  const sort = $derived(sortMode === 'updated' ? { key: 'updated', dir: sortDir } : null);
  function attemptText(job: Job): string {
    const n = attemptCount(job);
    if (n === 0) return '—';
    return `${n}/${job.max_attempts}`;
  }
</script>

<Table
  {columns}
  rows={jobs}
  key={(job) => job.job_id}
  selected={(job) => job.job_id === selectedId}
  off={(job) => boardColForState(job.state) === 'resolved'}
  {sort}
  onSort={onSortUpdated}
  ariaLabel="Jobs list"
  testid={TESTID.jobsTable}
>
  {#snippet cells(job)}
    {@const counts = acCounts(acRows(job))}
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
    <td class="c3 mono">{formatAgo(job.updated_at, now, 'bare') || '—'}</td>
  {/snippet}
</Table>

<style>
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
