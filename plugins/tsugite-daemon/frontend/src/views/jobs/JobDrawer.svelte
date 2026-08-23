<script lang="ts">
  // Job detail drawer (non-modal inspection panel). Renders the job's fields,
  // acceptance-criteria checklist, attempts stack, verifier reasoning, and the
  // links row; the footer offers only the mutations the backend allows from the
  // current state (cancel: any live state; mark-done: stuck; retry: stuck/errored).
  import Drawer from '$lib/components/overlays/Drawer.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import CheckItem from '$lib/components/rows/CheckItem.svelte';
  import Kv, { type KvItem } from '$lib/components/datadisplay/Kv.svelte';
  import { TESTID } from '$lib/testids';
  import type { Job } from '$lib/stores/jobs.svelte';
  import JobPill from './JobPill.svelte';
  import { acCounts, acRows, isTerminal } from './jobModel';
  import { formatAgo } from '$lib/relativeTime';
  import { attachRecordToChat, copyReference } from '../chats/attachRecord';

  let {
    job,
    now,
    terminalId = null,
    onClose,
    onRetry,
    onCancel,
    onMarkDone,
    onOpenChat,
    onOpenTerminal,
  }: {
    job: Job | null;
    now: number;
    /** Worker PTY id resolved by the view (null while probing / none exists). */
    terminalId?: string | null;
    onClose: () => void;
    onRetry: () => void;
    onCancel: () => void;
    onMarkDone: () => void;
    onOpenChat: (sessionId: string) => void;
    onOpenTerminal: (terminalId: string) => void;
  } = $props();

  const rows = $derived(job ? acRows(job) : []);
  const counts = $derived(acCounts(rows));
  const canRetry = $derived(job ? job.state === 'stuck' || job.state === 'errored' : false);
  const canMarkDone = $derived(job ? job.state === 'stuck' : false);
  const canCancel = $derived(
    job ? !isTerminal(job.state) || job.state === 'stuck' || job.state === 'errored' : false,
  );

  /* Verdict key for the attempt node; the key -> token mapping lives in CSS. */
  function attemptTone(pass: boolean | null): 'pass' | 'fail' | 'pending' {
    return pass === true ? 'pass' : pass === false ? 'fail' : 'pending';
  }
  function attemptVerdict(pass: boolean | null): string {
    return pass === true ? 'verified' : pass === false ? 'rejected' : 'in progress';
  }

  const details = $derived.by<KvItem[]>(() => {
    if (!job) return [];
    const items: KvItem[] = [{ term: 'agent', value: job.agent }];
    items.push({ term: 'executor', value: job.executor });
    if (job.model_ladder && job.model_ladder.length) {
      items.push({ term: 'model ladder', value: job.model_ladder.join(' → '), mono: true });
    } else if (job.model) {
      items.push({ term: 'model', value: job.model, mono: true });
    }
    if (job.verifier_model) items.push({ term: 'verifier', value: job.verifier_model, mono: true });
    if (job.effort) items.push({ term: 'effort', value: job.effort });
    if (job.repo) items.push({ term: 'repo', value: job.repo, mono: true });
    items.push({ term: 'notify', value: job.notify_when ?? 'never' });
    items.push({ term: 'created', value: formatAgo(job.created_at, now) || '—' });
    if (job.resolved_at) items.push({ term: 'resolved', value: formatAgo(job.resolved_at, now) });
    items.push({ term: 'job', value: job.job_id, mono: true });
    return items;
  });

  // Only an overall summary belongs here; per-criterion reasons already show as
  // CheckItem notes above, so this doesn't fall back to them (that double-renders).
  const reasoning = $derived.by(() => {
    if (!job) return '';
    const summary = job.result?.['summary'];
    if (typeof summary === 'string' && summary) return summary;
    const manual = job.result?.['manual_done_reason'];
    if (typeof manual === 'string' && manual) return manual;
    return '';
  });

  // parent/worker/verifier chat links share one chip renderer; absent ids drop
  // out. The worker-pty link differs (icon, handler) and stays separate below.
  const chatLinks = $derived.by(() => {
    const links: { id: string | null; testid: string; label: string }[] = [
      { id: job?.parent_session_id ?? null, testid: TESTID.jobLinkChat, label: 'parent chat' },
      { id: job?.worker_session_id ?? null, testid: TESTID.jobLinkWorker, label: 'worker' },
      { id: job?.verifier_session_id ?? null, testid: TESTID.jobLinkVerifier, label: 'verifier' },
    ];
    return links.filter((l): l is { id: string; testid: string; label: string } => l.id != null);
  });
</script>

<Drawer open={job !== null} title={job?.prompt ?? ''} label="Job detail" onclose={onClose}>
  {#snippet status()}
    {#if job}<JobPill st={job.state} />{/if}
  {/snippet}

  {#if job}
    <div data-testid={TESTID.jobDrawer} class="drawer-body">
      {#if job.state === 'awaiting_input' && job.pending_question}
        <div class="ask">
          <div class="ask-hd"><Icon name="q" />Worker question</div>
          <p class="ask-q">{job.pending_question}</p>
          <p class="ask-note">
            The worker is parked. Answer in the host chat and its agent resumes the job.
          </p>
          {#if job.parent_session_id}
            <Button variant="pri" size="sm" onclick={() => onOpenChat(job.parent_session_id!)}>
              {#snippet icon()}<Icon name="chat" />{/snippet}
              Answer in chat
            </Button>
          {/if}
        </div>
      {/if}

      {#if job.error && (job.state === 'errored' || job.state === 'stuck')}
        <div class="d-sec">
          <h4>failure</h4>
          <p class="err-msg">{job.error}</p>
          {#if job.error_detail}<pre class="err-detail">{job.error_detail}</pre>{/if}
        </div>
      {/if}

      {#if counts.total > 0}
        <div class="d-sec">
          <h4>acceptance criteria · {counts.pass}/{counts.total}</h4>
          {#each rows as row (row.index)}
            <CheckItem label={row.label} state={row.state} note={row.note} />
          {/each}
        </div>
      {/if}

      {#if job.attempts && job.attempts.length > 0}
        <div class="d-sec">
          <h4>attempts</h4>
          <div class="t-tl">
            {#each job.attempts as a, i (a.index)}
              <div
                class="t-tl-it"
                class:is-now={i === job.attempts.length - 1 && !isTerminal(job.state)}
              >
                <span class="nd"><i data-tone={attemptTone(a.verifier_pass)}></i></span>
                <span class="lb"
                  >attempt {a.index + 1} — {a.kind}
                  <span class="d"
                    >· {attemptVerdict(a.verifier_pass)}{a.model ? ` · ${a.model}` : ''}</span
                  ></span
                >
              </div>
            {/each}
          </div>
        </div>
      {/if}

      {#if reasoning}
        <div class="d-sec">
          <h4>verifier reasoning</h4>
          <p class="reason">{reasoning}</p>
        </div>
      {/if}

      <div class="d-sec">
        <h4>details</h4>
        <Kv items={details} />
      </div>

      {#if chatLinks.length > 0 || terminalId}
        <div class="d-sec">
          <h4>links</h4>
          <div class="links">
            {#each chatLinks as link (link.testid)}
              <button
                type="button"
                class="t-chip linkchip"
                data-testid={link.testid}
                onclick={() => onOpenChat(link.id)}
              >
                <Icon name="chat" />{link.label}<Icon name="out" size={9} />
              </button>
            {/each}
            {#if terminalId}
              <button
                type="button"
                class="t-chip linkchip"
                data-testid={TESTID.jobLinkTerminal}
                onclick={() => onOpenTerminal(terminalId!)}
              >
                <Icon name="term" />worker pty<Icon name="out" size={9} />
              </button>
            {/if}
          </div>
        </div>
      {/if}
    </div>
  {/if}

  {#snippet footer()}
    {#if job}
      <Button
        size="sm"
        variant="ghost"
        onclick={() => {
          if (job) void attachRecordToChat('job', job.job_id);
        }}
      >
        {#snippet icon()}<Icon name="chat" />{/snippet}
        Add to chat
      </Button>
      <Button
        size="sm"
        variant="ghost"
        onclick={() => {
          if (job) void copyReference('job', job.job_id);
        }}
      >
        {#snippet icon()}<Icon name="link" />{/snippet}
        Copy reference
      </Button>
      {#if job.state === 'awaiting_input' && job.parent_session_id}
        <Button variant="pri" size="sm" onclick={() => onOpenChat(job.parent_session_id!)}>
          {#snippet icon()}<Icon name="chat" />{/snippet}
          Answer in chat
        </Button>
      {/if}
      {#if canRetry}
        <Button size="sm" data-testid={TESTID.jobRetry} onclick={onRetry}>
          {#snippet icon()}<Icon name="retry" />{/snippet}
          Retry
        </Button>
      {/if}
      {#if canMarkDone}
        <Button size="sm" data-testid={TESTID.jobMarkDone} onclick={onMarkDone}>
          {#snippet icon()}<Icon name="check" />{/snippet}
          Mark done
        </Button>
      {/if}
      {#if canCancel}
        <div class="grow"></div>
        <Button variant="danger" size="sm" data-testid={TESTID.jobCancel} onclick={onCancel}>
          Cancel
        </Button>
      {/if}
    {/if}
  {/snippet}
</Drawer>

<style>
  .drawer-body {
    display: grid;
    gap: 16px;
    align-content: start;
  }
  .d-sec > h4 {
    margin: 0 0 7px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  /* worker-question block - the .t-ask contract, read-only (answering routes to
     the host chat: jobs have no direct HTTP answer endpoint). */
  .ask {
    border: 1px solid color-mix(in oklab, var(--st-warn) 45%, transparent);
    background: color-mix(in oklab, var(--st-warn) 8%, transparent);
    border-radius: var(--r-lg);
    padding: 11px 13px;
    display: grid;
    gap: 9px;
  }
  .ask-hd {
    display: flex;
    align-items: center;
    gap: 7px;
    font: 600 var(--fs-sm) / 1 var(--font-ui);
    color: var(--st-warn);
  }
  .ask-q {
    margin: 0;
    font-size: var(--fs-md);
    color: var(--tx0);
    line-height: 1.5;
    text-wrap: pretty;
  }
  .ask-note {
    margin: 0;
    font: 400 var(--fs-xs) / 1.45 var(--font-ui);
    color: var(--tx3);
  }
  .err-msg {
    margin: 0;
    font: 500 var(--fs-sm) var(--font-mono);
    color: var(--st-err);
  }
  .err-detail {
    margin: 6px 0 0;
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
    color: var(--tx2);
    padding: 8px 10px;
    max-height: 160px;
    overflow: auto;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .reason {
    margin: 0;
    color: var(--tx2);
    font-size: var(--fs-xs);
    line-height: 1.6;
    max-width: 64ch;
  }
  /* timeline */
  .t-tl {
    display: grid;
    gap: 0;
    font-size: var(--fs-xs);
  }
  .t-tl-it {
    display: grid;
    grid-template-columns: 14px 1fr auto;
    gap: 0 9px;
    position: relative;
    padding: 0 0 12px;
  }
  .t-tl-it::before {
    content: '';
    position: absolute;
    left: 6.5px;
    top: 13px;
    bottom: 1px;
    width: 1px;
    background: var(--bd1);
  }
  .t-tl-it:last-child::before {
    display: none;
  }
  .t-tl-it .nd {
    width: 14px;
    height: 14px;
    display: grid;
    place-items: center;
    margin-top: 1px;
  }
  .t-tl-it .nd i {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--c, var(--bd1));
    display: block;
  }
  .t-tl-it .nd i[data-tone='pass'] {
    --c: var(--st-ok);
  }
  .t-tl-it .nd i[data-tone='fail'] {
    --c: var(--st-err);
  }
  .t-tl-it .nd i[data-tone='pending'] {
    --c: var(--st-warn);
  }
  .t-tl-it.is-now .nd i {
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--c, var(--st-warn)) 22%, transparent);
  }
  .t-tl-it .lb {
    color: var(--tx1);
    font-family: var(--font-mono);
  }
  .t-tl-it .lb .d {
    color: var(--tx3);
  }
  .links {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
  }
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
  .linkchip {
    cursor: pointer;
  }
  .linkchip:hover {
    border-color: var(--acc);
    color: var(--acc);
  }
  .linkchip:hover :global(.ic) {
    color: var(--acc);
  }
  .linkchip:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .grow {
    flex: 1;
  }
</style>
