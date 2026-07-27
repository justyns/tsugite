<script lang="ts">
  // Board card (.t-job). A real button: click / Enter opens the detail drawer.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';
  import type { Job } from '$lib/stores/jobs.svelte';
  import JobPill from './JobPill.svelte';
  import JobCrit from './JobCrit.svelte';
  import { acCounts, acRows, attemptCount } from './jobModel';
  import { relativeAgo, relativeTime } from './format';
  import { boardColForState } from './board';

  let {
    job,
    now,
    selected = false,
    onOpen,
  }: {
    job: Job;
    /** Clock tick (ms) so relative times refresh live. */
    now: number;
    selected?: boolean;
    onOpen: () => void;
  } = $props();

  const col = $derived(boardColForState(job.state));
  const counts = $derived(acCounts(acRows(job)));
  const attempts = $derived(attemptCount(job));

  // Secondary line under the pill: attempt / queue-age / resolved-age, per state.
  const meta = $derived.by(() => {
    if (col === 'queued') return `${relativeTime(job.created_at, now)} in queue`;
    if (job.state === 'done')
      return `${relativeAgo(job.resolved_at ?? job.updated_at, now)} · ${attempts} attempt${attempts === 1 ? '' : 's'}`;
    if (job.state === 'cancelled') return relativeAgo(job.resolved_at ?? job.updated_at, now);
    return `attempt ${attempts}/${job.max_attempts}`;
  });
  // Trailing timestamp for the live/parked states (queued/done/cancelled carry
  // their timing in `meta` already).
  const stamp = $derived.by(() => {
    if (col === 'queued' || job.state === 'done' || job.state === 'cancelled') return '';
    if (job.state === 'awaiting_input') return `blocked ${relativeTime(job.updated_at, now)}`;
    return relativeAgo(job.updated_at, now);
  });
  const errorLine = $derived(
    job.state === 'errored' || job.state === 'stuck' ? (job.error ?? '') : '',
  );
</script>

<button
  type="button"
  class="t-job"
  class:t-job--attn={col === 'needs-you'}
  class:is-selected={selected}
  data-testid={TESTID.jobCard(job.job_id)}
  aria-pressed={selected}
  onclick={onOpen}
>
  <span class="jt">{job.prompt}</span>
  <span class="jm">
    <JobPill st={job.state} />
    {#if meta}<span>{meta}</span>{/if}
    {#if stamp}<span class="mono">{stamp}</span>{/if}
  </span>
  {#if job.state === 'awaiting_input' && job.pending_question}
    <span class="qprev">{job.pending_question}</span>
  {/if}
  {#if errorLine}
    <span class="jm mono errline" class:stuck={job.state === 'stuck'}>{errorLine}</span>
  {/if}
  <span class="jm">
    <span class="t-chip"><Icon name="agent" />{job.agent}</span>
    {#if job.executor && job.executor !== 'agent'}
      <span class="t-chip"><Icon name="term" />{job.executor}</span>
    {/if}
    <JobCrit {counts} showWord={col === 'queued'} />
  </span>
</button>

<style>
  .t-job {
    background: var(--bg2);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 9px 10px;
    display: grid;
    gap: 7px;
    cursor: pointer;
    text-align: left;
    font-family: inherit;
    color: inherit;
    min-width: 0;
    transition: border-color var(--t-1);
  }
  .t-job:hover {
    border-color: var(--bd1);
    background: color-mix(in oklab, var(--bg3) 55%, var(--bg2));
  }
  .t-job:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .t-job.is-selected {
    border-color: var(--acc);
    box-shadow: 0 0 0 1px var(--acc);
  }
  .t-job--attn {
    box-shadow: inset 2px 0 0 0 var(--st-warn);
  }
  .t-job--attn.is-selected {
    box-shadow:
      inset 2px 0 0 0 var(--st-warn),
      0 0 0 1px var(--acc);
  }
  .jt {
    font: 500 var(--fs-sm) / 1.4 var(--font-ui);
    color: var(--tx0);
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  .jm {
    display: flex;
    align-items: center;
    gap: 7px;
    flex-wrap: wrap;
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .errline {
    color: var(--st-err);
  }
  .errline.stuck {
    color: var(--st-warn);
  }
  .qprev {
    font: 400 var(--fs-xs) / 1.45 var(--font-mono);
    color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 9%, transparent);
    border-radius: var(--r-sm);
    padding: 5px 7px;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
  }
  /* The chip's own `.ic` colour is set by the shared Chip CSS; these cards use a
     bare `.t-chip` span, so restate the icon tint here. */
  .t-chip {
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
  }
  .t-chip :global(.ic) {
    width: 10px;
    height: 10px;
    color: var(--tx3);
  }
</style>
