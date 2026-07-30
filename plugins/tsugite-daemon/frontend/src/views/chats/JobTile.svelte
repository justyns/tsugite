<script lang="ts">
  // Compact in-chat job tile: a job_status timeline block rendered as a small
  // card. The block is a SPAWN-TIME snapshot; the live jobs store (fed by
  // job_update broadcasts app-wide) is the source of truth for the current
  // state, so a finished job stops reading "running" here. "Open" both navigates
  // to the Jobs view and asks it to open this job's drawer.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import { jobs } from '$lib/stores/jobs.svelte';
  import { jobDrawerRequest } from '../jobs/jobDrawerSignal.svelte';
  import { attachRecordToChat, copyReference } from './attachRecord';
  import type { JobLike } from '$lib/stores/jobsFilter';

  let { job }: { job: JobLike } = $props();

  // Labels + spin here; each state's color lives in the stylesheet below.
  const STATE_META: Record<string, { label: string; spin?: boolean }> = {
    queued: { label: 'queued' },
    running: { label: 'running', spin: true },
    verifying: { label: 'verifying', spin: true },
    awaiting_input: { label: 'awaiting input' },
    stuck: { label: 'stuck' },
    errored: { label: 'errored' },
    done: { label: 'done' },
    cancelled: { label: 'cancelled' },
  };

  const jobId = $derived(job.job_id ?? '');
  // Prefer the live store record over the recorded snapshot when the store has it.
  const live = $derived(jobId ? jobs.jobs.find((j) => j.job_id === jobId) : undefined);
  const j = $derived(live ?? job);

  const state = $derived(j.state ?? 'queued');
  const meta = $derived(STATE_META[state] ?? { label: state });
  const prompt = $derived(j.prompt ?? '');
  const attempts = $derived(
    j.verify_attempts != null && j.max_attempts != null
      ? `attempt ${j.verify_attempts}/${j.max_attempts}`
      : '',
  );
  const agent = $derived(j.agent ?? '');

  function openInJobs() {
    if (jobId) jobDrawerRequest.request(jobId);
  }
</script>

<div class="jobtile" data-st={state}>
  <span class="ind"><Icon name="jobs" size={13} /></span>
  <div class="body">
    <div class="top">
      <span class="state" data-k={state}>
        {#if meta.spin}<Spin />{:else}<span class="dot" aria-hidden="true"></span>{/if}
        {meta.label}
      </span>
      {#if attempts}<span class="mono">{attempts}</span>{/if}
      {#if agent}<span class="chip"><Icon name="agent" size={10} />{agent}</span>{/if}
    </div>
    {#if prompt}<div class="prompt">{prompt}</div>{/if}
  </div>
  <div class="acts">
    <button
      class="act"
      type="button"
      aria-label="Add job to chat"
      onclick={() => void attachRecordToChat('job', jobId)}
    >
      <Icon name="chat" size={12} />
    </button>
    <button
      class="act"
      type="button"
      aria-label="Copy job reference"
      onclick={() => void copyReference('job', jobId)}
    >
      <Icon name="link" size={12} />
    </button>
    <a class="open" href="#jobs" onclick={openInJobs} aria-label="Open in Jobs">
      <Icon name="out" size={12} />open
    </a>
  </div>
</div>

<style>
  .jobtile {
    display: flex;
    align-items: flex-start;
    gap: 9px;
    padding: 8px 10px;
    background: var(--bg2);
    border: 1px solid var(--bd0);
    border-left: 2px solid var(--c, var(--bd1));
    border-radius: var(--r-md);
    min-width: 0;
  }
  .jobtile[data-st='awaiting_input'],
  .jobtile[data-st='stuck'] {
    --c: var(--st-warn);
  }
  .jobtile[data-st='errored'] {
    --c: var(--st-err);
  }
  .jobtile[data-st='running'],
  .jobtile[data-st='verifying'] {
    --c: var(--st-ok);
  }
  .ind {
    color: var(--tx3);
    padding-top: 2px;
    flex: none;
  }
  .body {
    flex: 1;
    min-width: 0;
    display: grid;
    gap: 3px;
  }
  .top {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
  }
  .state {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 600 var(--fs-xs) var(--font-mono);
    color: var(--c, var(--st-mute));
  }
  /* Per-state colors (scoped here: the tile root's --c only tints the border
     for live/attention states, while the label always names its state). */
  .state[data-k='queued'] {
    --c: var(--st-queue);
  }
  .state[data-k='running'],
  .state[data-k='done'] {
    --c: var(--st-ok);
  }
  .state[data-k='verifying'] {
    --c: var(--st-info);
  }
  .state[data-k='awaiting_input'],
  .state[data-k='stuck'] {
    --c: var(--st-warn);
  }
  .state[data-k='errored'] {
    --c: var(--st-err);
  }
  .state[data-k='cancelled'] {
    --c: var(--st-mute);
  }
  .state .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--c);
    display: inline-block;
  }
  .state :global(.t-spin) {
    color: var(--c);
  }
  .mono {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .chip {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx2);
  }
  .prompt {
    font-size: var(--fs-sm);
    color: var(--tx1);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .acts {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: none;
    align-self: center;
  }
  .act {
    display: inline-flex;
    align-items: center;
    padding: 2px;
    border: 0;
    background: none;
    color: var(--tx3);
    cursor: pointer;
  }
  .act:hover {
    color: var(--acc);
  }
  .open {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    flex: none;
    align-self: center;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    text-decoration: none;
  }
  .open:hover {
    color: var(--acc);
  }
</style>
