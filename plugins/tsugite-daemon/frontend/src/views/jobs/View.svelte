<script lang="ts">
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Select from '$lib/components/inputs/Select.svelte';
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import Modal from '$lib/components/overlays/Modal.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import { TESTID } from '$lib/testids';
  import { api } from '$lib/api/client';
  import { auth } from '$lib/stores/auth.svelte';
  import { jobs, type Job, type JobRetryOpts } from '$lib/stores/jobs.svelte';
  import { jobDrawerRequest } from './jobDrawerSignal.svelte';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { navigate } from '$lib/router.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import {
    ATTENTION_COL,
    BOARD_COL_LABEL,
    FILTER_KEYS,
    type FilterKey,
    SORT_LABEL,
    SORT_MODES,
    type SortMode,
    applyColumnFilter,
    filterCounts,
    sortJobs,
  } from './board';
  import JobBoard from './JobBoard.svelte';
  import JobList from './JobList.svelte';
  import JobDrawer from './JobDrawer.svelte';
  import JobRetryModal from './JobRetryModal.svelte';
  import NewJobDrawer, { type NewJobForm } from './NewJobDrawer.svelte';

  let layout = $state<'board' | 'list'>('board');
  let filterKey = $state<FilterKey>('all');
  let sortLabel = $state(SORT_LABEL.urgency);
  let sortDir = $state<'ascending' | 'descending'>('descending');
  let selectedId = $state<string | null>(null);
  let retryOpen = $state(false);
  let newJobOpen = $state(false);
  let submitting = $state(false);
  let confirm = $state<{ kind: 'cancel' | 'markDone'; jobId: string } | null>(null);
  let now = $state(Date.now());

  const SORT_OPTIONS = SORT_MODES.map((m) => SORT_LABEL[m]);
  const sortMode = $derived<SortMode>(
    SORT_MODES.find((m) => SORT_LABEL[m] === sortLabel) ?? 'urgency',
  );

  const searched = $derived(jobs.filtered);
  const counts = $derived(filterCounts(searched));
  const ordered = $derived.by(() => {
    const s = sortJobs(applyColumnFilter(searched, filterKey), sortMode);
    return sortMode === 'updated' && sortDir === 'ascending' ? s.reverse() : s;
  });

  const selectedJob = $derived<Job | null>(
    selectedId ? (jobs.jobs.find((j) => j.job_id === selectedId) ?? null) : null,
  );

  // Resolve the worker PTY for the open job (direct id, else parent-session probe).
  let terminalId = $state<string | null>(null);
  $effect(() => {
    const job = selectedJob;
    terminalId = null;
    if (!job) return;
    let live = true;
    void jobs.terminalIdForJob(job).then((id) => {
      if (live) terminalId = id;
    });
    return () => {
      live = false;
    };
  });

  // Live clock so relative times / queue ages tick without a refetch.
  $effect(() => {
    const t = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(t);
  });

  $effect(() => {
    void jobs.load();
    void jobs.loadExecutors();
    void agentsMeta.load();
  });

  // An in-chat job tile's "open" lands here (via #jobs) with a pending request:
  // open that job's drawer once it is present in the store (the initial load may
  // still be in flight, so this re-runs when jobs.jobs fills).
  $effect(() => {
    const id = jobDrawerRequest.pending;
    if (!id) return;
    const job = jobs.jobs.find((j) => j.job_id === id);
    if (job) {
      openJob(job);
      jobDrawerRequest.consume();
    }
  });

  const agentNames = $derived(agentsMeta.agents.map((a) => a.name));

  // The detail drawer and the composer share the one right-side slot, so opening
  // either closes the other.
  function openJob(job: Job) {
    newJobOpen = false;
    selectedId = job.job_id;
  }
  function closeDrawer() {
    selectedId = null;
  }
  function openNewJob() {
    selectedId = null;
    newJobOpen = true;
  }
  function onSortUpdated() {
    if (sortMode === 'updated') {
      sortDir = sortDir === 'descending' ? 'ascending' : 'descending';
    } else {
      sortLabel = SORT_LABEL.updated;
      sortDir = 'descending';
    }
  }

  // Route via the hash so the shell also LEAVES this full-area view: a bare
  // spaces.openReusing would retarget a tab inside the hidden workspace and the
  // click would look like a no-op. The deep-link effect activates the view and
  // opens the surface.
  //
  // No agent param: job.agent is the WORKER agent file (e.g. job_worker), never a
  // chat adapter, and it applies to none of the three links anyway - parent,
  // worker, and verifier sessions each run under the parent's agent. The chat
  // surface resolves each session's true agent from its record.
  function openChat(sessionId: string) {
    navigate('chats', { sessionId });
  }
  function openTerminal(id: string) {
    navigate('terminals', { terminalId: id });
  }

  function errMsg(e: unknown): string {
    return e instanceof Error ? e.message : String(e);
  }

  // Focus management for the right-side slot. Only one drawer is mounted at a
  // time (they share the slot), so the Drawer's own closed->open focus never
  // fires; this moves focus into the drawer on open and restores it on close.
  // `preventScroll` is load-bearing: a plain focus() scrolls the surface
  // sideways to reveal the still-sliding drawer, dragging everything with it.
  function drawerFocus(node: HTMLElement) {
    const restoreTo = document.activeElement as HTMLElement | null;
    const target = node.querySelector<HTMLElement>('button, [href], input, textarea, select');
    target?.focus({ preventScroll: true });
    return {
      destroy() {
        restoreTo?.focus?.({ preventScroll: true });
      },
    };
  }

  async function doRetry(opts: JobRetryOpts) {
    const id = selectedId;
    if (!id) return;
    try {
      await jobs.retry(id, opts);
      retryOpen = false;
      toasts.push('ok', 'Retrying job', { icon: 'check' });
      await jobs.load();
    } catch (e) {
      toasts.push('err', 'Retry failed', { body: errMsg(e) });
    }
  }

  async function runConfirm() {
    if (!confirm) return;
    const { kind, jobId } = confirm;
    try {
      if (kind === 'cancel') {
        await jobs.cancel(jobId);
        toasts.push('ok', 'Job cancelled');
      } else {
        await jobs.markDone(jobId);
        toasts.push('ok', 'Job marked done');
      }
      confirm = null;
      await jobs.load();
    } catch (e) {
      toasts.push('err', kind === 'cancel' ? 'Cancel failed' : 'Mark-done failed', {
        body: errMsg(e),
      });
    }
  }

  async function submitNewJob(form: NewJobForm) {
    submitting = true;
    try {
      const body: Record<string, unknown> = {
        user_id: auth.userId,
        prompt: form.prompt,
        max_attempts: form.maxAttempts,
      };
      if (form.acceptanceCriteria.length)
        body.acceptance_criteria = form.acceptanceCriteria.join('|');
      if (form.executor && form.executor !== 'agent') body.executor = form.executor;
      if (form.notifyWhen && form.notifyWhen !== 'never') body.notify_when = form.notifyWhen;
      const res = await api.post<{ result?: string }>(
        `/api/agents/${encodeURIComponent(form.agent)}/commands/job`,
        body,
      );
      newJobOpen = false;
      toasts.push('ok', 'Job spawned', res.result ? { body: res.result } : {});
      await jobs.load();
    } catch (e) {
      toasts.push('err', 'Could not spawn job', { body: errMsg(e) });
    } finally {
      submitting = false;
    }
  }

  const confirmTitle = $derived(confirm?.kind === 'markDone' ? 'Mark job done?' : 'Cancel job?');
</script>

<section class="jobs" data-testid={TESTID.view('jobs')}>
  <div class="jobs-tools">
    <span data-testid={TESTID.jobsLayout}>
      <Seg options={['board', 'list']} bind:value={layout} ariaLabel="Layout" />
    </span>
    <div class="fpills" role="group" aria-label="Filter jobs">
      {#each FILTER_KEYS as key (key)}
        <button
          type="button"
          class="fpill"
          class:is-active={filterKey === key}
          class:fpill--attn={key === ATTENTION_COL}
          aria-pressed={filterKey === key}
          data-testid={TESTID.jobsFilter(key)}
          onclick={() => (filterKey = key)}
        >
          {key === 'all' ? 'all' : BOARD_COL_LABEL[key]}
          <span class="n">{counts[key]}</span>
        </button>
      {/each}
    </div>
    <div class="grow"></div>
    <span class="searchw" data-testid={TESTID.jobsSearch}>
      <SearchInput
        bind:value={jobs.filterText}
        ariaLabel="Search jobs"
        placeholder="search jobs…"
        shortcutKey="f"
      />
    </span>
    <span data-testid={TESTID.jobsSort}>
      <Select options={SORT_OPTIONS} bind:value={sortLabel} ariaLabel="Sort" />
    </span>
    <Button variant="pri" size="sm" data-testid={TESTID.jobsNew} onclick={openNewJob}>
      {#snippet icon()}<Icon name="plus" />{/snippet}
      New job
    </Button>
  </div>

  <div class="jobs-view" class:drawer-open={selectedJob !== null || newJobOpen}>
    {#if jobs.loading && jobs.jobs.length === 0}
      <div class="state"><Spin /> <span>loading jobs…</span></div>
    {:else if jobs.error}
      <div class="state err">
        <Icon name="alert" /> <span>{jobs.error}</span>
        <Button size="sm" onclick={() => jobs.load()}>Retry</Button>
      </div>
    {:else if ordered.length === 0}
      <div class="state" data-testid={TESTID.jobsEmpty}>
        <Icon name="jobs" />
        <span>{jobs.jobs.length === 0 ? 'No jobs yet.' : 'No jobs match this filter.'}</span>
        {#if jobs.jobs.length === 0}
          <Button variant="pri" size="sm" onclick={openNewJob}>
            {#snippet icon()}<Icon name="plus" />{/snippet}
            New job
          </Button>
        {/if}
      </div>
    {:else if layout === 'board'}
      <div class="boardwrap">
        <JobBoard jobs={ordered} {now} {selectedId} onOpen={openJob} />
      </div>
    {:else}
      <div class="listwrap">
        <JobList
          jobs={ordered}
          {now}
          {selectedId}
          {sortMode}
          {sortDir}
          onOpen={openJob}
          {onSortUpdated}
        />
      </div>
    {/if}
  </div>

  <!-- The detail drawer and the composer share the one right-side slot; only one
       is mounted at a time so a closed sibling never leaves scroll room for the
       on-open focus to drag the surface into. -->
  {#if selectedJob}
    <div use:drawerFocus class="drawer-mount">
      <JobDrawer
        job={selectedJob}
        {now}
        {terminalId}
        onClose={closeDrawer}
        onRetry={() => (retryOpen = true)}
        onCancel={() => selectedId && (confirm = { kind: 'cancel', jobId: selectedId })}
        onMarkDone={() => selectedId && (confirm = { kind: 'markDone', jobId: selectedId })}
        onOpenChat={openChat}
        onOpenTerminal={openTerminal}
      />
    </div>
  {/if}

  {#if newJobOpen}
    <div use:drawerFocus class="drawer-mount">
      <NewJobDrawer
        open={true}
        agents={agentNames}
        executors={jobs.executors}
        {submitting}
        onClose={() => (newJobOpen = false)}
        onSubmit={submitNewJob}
      />
    </div>
  {/if}

  <JobRetryModal
    open={retryOpen}
    prompt={selectedJob?.prompt ?? ''}
    onClose={() => (retryOpen = false)}
    onSubmit={doRetry}
  />

  <Modal
    open={confirm !== null}
    title={confirmTitle}
    tone={confirm?.kind === 'cancel' ? 'danger' : 'default'}
    onclose={() => (confirm = null)}
  >
    <div data-testid={TESTID.jobCancelModal}>
      {#if confirm?.kind === 'markDone'}
        Override this stuck job to <code>done</code>? It records a manual success without passing
        verification.
      {:else}
        Stop this job? Any in-flight worker is cancelled. This cannot be undone.
      {/if}
    </div>
    {#snippet footer()}
      <Button variant="ghost" size="sm" onclick={() => (confirm = null)}>Keep it</Button>
      <Button
        variant={confirm?.kind === 'cancel' ? 'danger' : 'pri'}
        size="sm"
        data-autofocus
        onclick={runConfirm}
      >
        {confirm?.kind === 'markDone' ? 'Mark done' : 'Cancel job'}
      </Button>
    {/snippet}
  </Modal>
</section>

<style>
  .jobs {
    position: relative;
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  /* Transparent to layout: the focus-managing wrapper exists only to give the
     use:action a mount/unmount hook; the Drawer inside positions against the
     .jobs section as usual. */
  .drawer-mount {
    display: contents;
  }
  .jobs-tools {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    flex-wrap: wrap;
    flex: none;
  }
  .fpills {
    display: flex;
    gap: 4px;
    flex-wrap: wrap;
  }
  .fpill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
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
  .fpill .n {
    color: var(--tx3);
    font-size: var(--fs-2xs);
  }
  .fpill.is-active .n {
    color: var(--tx1);
  }
  .fpill--attn.is-active {
    background: color-mix(in oklab, var(--st-warn) 15%, transparent);
    border-color: color-mix(in oklab, var(--st-warn) 45%, transparent);
    color: var(--st-warn);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .searchw {
    width: 190px;
    max-width: 40vw;
  }
  .jobs-view {
    flex: 1;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: margin-right var(--t-3) var(--ease);
  }
  /* The detail drawer overlays the right edge; shrink the board area so every
     column (incl. resolved) can still scroll into view beside it. Narrow frames
     keep the full-overlay behavior (the drawer covers everything there). */
  @media (min-width: 701px) {
    .jobs-view.drawer-open {
      margin-right: min(480px, 60%);
    }
  }
  .boardwrap {
    flex: 1;
    min-height: 0;
  }
  .listwrap {
    flex: 1;
    min-height: 0;
    overflow: auto;
  }
  .state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    color: var(--tx3);
    font-size: var(--fs-sm);
    padding: var(--sp-5);
    text-align: center;
  }
  .state.err {
    color: var(--st-err);
  }

  /* Narrow: compact tools, search flexes, the sort select sheds (the board
     stacks vertically so sort matters less than reach). */
  @media (max-width: 640px) {
    .jobs-tools {
      gap: 6px;
      padding: 8px 10px;
    }
    .searchw {
      width: auto;
      flex: 1 1 130px;
      max-width: none;
    }
    .jobs-tools [data-testid='jobs-sort'] {
      display: none;
    }
  }
</style>
