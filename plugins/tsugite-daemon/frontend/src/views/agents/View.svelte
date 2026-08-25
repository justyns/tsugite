<script lang="ts">
  // Agents builder: a master roster of agent `.md` files (left) and a
  // form / markdown editor (right). The Markdown tab is the editable source of
  // truth (PUT /api/agent-files/content); Form is a parsed structured reflection
  // of the frontmatter. The Run launcher opens a one-shot chat surface bound to
  // the agent. Roster reconciles two backend lists: the editable file set
  // (/api/agent-files) enriched with the registered-agent roster (/api/agents),
  // which carries running_tasks and marks which files are live runnable agents.
  import { onMount, untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import { agentsMeta, type MdFile } from '$lib/stores/agentsMeta.svelte';
  import { spaces } from '$lib/stores/spaces.svelte';
  import { navigate } from '$lib/router.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { nextRovingIndex } from '$lib/actions/rovingNav';
  import { parseAgentFile, summarizeAgent } from './agentFrontmatter';
  import AgentForm from './AgentForm.svelte';
  import RunLauncher from './RunLauncher.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Badge from '$lib/components/buttons/Badge.svelte';
  import Dot from '$lib/components/buttons/Dot.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';

  type Mode = 'form' | 'markdown';

  let selectedPath = $state<string | null>(null);
  let content = $state('');
  let savedContent = $state('');
  let mode = $state<Mode>('form');
  let query = $state('');

  let loadingContent = $state(false);
  let contentError = $state<string | null>(null);
  let saving = $state(false);
  let justSaved = $state(false);
  let launcherOpen = $state(false);

  let contentSeq = 0;

  const msg = (e: unknown) => (e instanceof Error ? e.message : String(e));

  const runningAgentFile = $derived(agentsMeta.runtime?.agent_file ?? null);

  function rankOf(f: MdFile): number {
    if (f.name === runningAgentFile) return 0;
    return f.readonly ? 2 : 1;
  }

  const sortedFiles = $derived(
    [...agentsMeta.agentFiles].sort(
      (a, b) => rankOf(a) - rankOf(b) || a.name.localeCompare(b.name),
    ),
  );

  const visibleFiles = $derived(
    query.trim() === ''
      ? sortedFiles
      : sortedFiles.filter((f) => {
          const q = query.toLowerCase();
          return (
            f.name.toLowerCase().includes(q) ||
            f.description.toLowerCase().includes(q) ||
            f.source.toLowerCase().includes(q)
          );
        }),
  );

  const selectedVisible = $derived(visibleFiles.some((f) => f.path === selectedPath));
  const selectedFile = $derived(agentsMeta.agentFiles.find((f) => f.path === selectedPath) ?? null);
  const readonly = $derived(selectedFile?.readonly ?? false);
  const parsed = $derived(parseAgentFile(content));
  const summary = $derived(summarizeAgent(parsed.frontmatter));
  const agentName = $derived(summary.name ?? selectedFile?.name ?? '');
  const registered = $derived(agentName !== '' && agentName === runningAgentFile);
  const runningTasks = $derived(registered ? (agentsMeta.runtime?.running_tasks ?? 0) : 0);
  const dirty = $derived(content !== savedContent);

  const sourceLabel = $derived(selectedFile ? selectedFile.source : '');

  onMount(() => {
    void agentsMeta.load();
    void agentsMeta.loadAgentFiles();
  });

  // Auto-select the top-ranked file once the list is available.
  $effect(() => {
    if (!selectedPath && sortedFiles.length > 0) {
      const first = sortedFiles[0]!.path;
      untrack(() => selectFile(first));
    }
  });

  async function loadContent(path: string): Promise<void> {
    const seq = ++contentSeq;
    loadingContent = true;
    contentError = null;
    try {
      const r = await agentsMeta.readAgentFile(path);
      if (seq !== contentSeq) return;
      content = r.content;
      savedContent = r.content;
    } catch (e) {
      if (seq === contentSeq) contentError = msg(e);
    } finally {
      if (seq === contentSeq) loadingContent = false;
    }
  }

  function selectFile(path: string): void {
    if (path === selectedPath) return;
    selectedPath = path;
    void loadContent(path);
  }

  async function save(): Promise<void> {
    if (!selectedFile || readonly || !dirty || saving) return;
    saving = true;
    try {
      await agentsMeta.saveAgentFile(selectedFile.path, content);
      savedContent = content;
      justSaved = true;
      setTimeout(() => (justSaved = false), 2500);
      toasts.push('ok', 'Saved', { body: `${agentName} updated` });
      await agentsMeta.loadAgentFiles();
    } catch (e) {
      toasts.push('err', 'Save failed', { body: msg(e) });
    } finally {
      saving = false;
    }
  }

  function launch(opts: { prompt: string; effort?: string }): void {
    const params: Record<string, string> = { agent: agentName, prompt: opts.prompt };
    if (opts.effort) params.effort = opts.effort;
    spaces.open({ kind: 'chat', params, title: `Run: ${agentName}` });
    launcherOpen = false;
    // This is a full-area view: the tab just docked into the HIDDEN workspace,
    // so switch to it - otherwise the launch looks like a no-op.
    navigate('chats');
    toasts.push('ok', 'Chat started', { body: `${agentName} · new session` });
  }

  function onRowKeydown(e: KeyboardEvent, index: number): void {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const f = visibleFiles[index];
      if (f) selectFile(f.path);
      return;
    }
    const next = nextRovingIndex(index, e.key, visibleFiles.length);
    if (next === null) return;
    e.preventDefault();
    const host = e.currentTarget as HTMLElement;
    const rows = host.closest('[role="listbox"]')?.querySelectorAll<HTMLElement>('[data-rowidx]');
    rows?.[next]?.focus();
  }

  function onEditorKeydown(e: KeyboardEvent): void {
    if ((e.metaKey || e.ctrlKey) && (e.key === 's' || e.key === 'S')) {
      e.preventDefault();
      void save();
    }
  }

  function metaFor(): string {
    const parts: string[] = [];
    if (sourceLabel) parts.push(sourceLabel);
    parts.push(registered ? 'registered' : 'not registered');
    if (readonly) parts.push('read-only');
    return parts.join(' · ');
  }
</script>

<section class="agents-shell" data-testid={TESTID.view('agents')}>
  <div class="agents-list" data-testid={TESTID.agentsRoster}>
    <div class="sidebar-hd">
      <strong class="hd-title">Agents</strong>
      <span class="cnt">{visibleFiles.length}</span>
    </div>
    <div class="sidebar-search">
      <SearchInput bind:value={query} ariaLabel="Filter agents" placeholder="filter agents…" />
    </div>

    {#if agentsMeta.loading && agentsMeta.agentFiles.length === 0}
      <div class="pad"><PaneState kind="loading" lines={6} /></div>
    {:else if agentsMeta.error}
      <div class="pad">
        <PaneState kind="error" title="Couldn't load agents">
          {#snippet hint()}<span>{agentsMeta.error}</span>{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={() => void agentsMeta.loadAgentFiles()}>Retry</Button>
          {/snippet}
        </PaneState>
      </div>
    {:else if visibleFiles.length === 0}
      <div class="pad">
        <PaneState kind="empty" title={query ? 'No matches' : 'No agents'}>
          {#snippet hint()}
            <span
              >{query
                ? 'No agent file matches your filter.'
                : 'No agent files were discovered.'}</span
            >
          {/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="roster" role="listbox" aria-label="Agents" tabindex="-1">
        {#each visibleFiles as f, i (f.path)}
          {@const isSel = f.path === selectedPath}
          {@const live = f.name === runningAgentFile}
          {@const running = live ? (agentsMeta.runtime?.running_tasks ?? 0) : 0}
          <button
            type="button"
            class="ag-row"
            class:is-selected={isSel}
            role="option"
            aria-selected={isSel}
            tabindex={isSel || (!selectedVisible && i === 0) ? 0 : -1}
            data-rowidx={i}
            data-testid={TESTID.agentRow(f.name)}
            onclick={() => selectFile(f.path)}
            onkeydown={(e) => onRowKeydown(e, i)}
          >
            <span class="ag-dot"><Dot color={live ? 'ok' : 'mute'} ring={!live} /></span>
            <span class="ag-main">
              <span class="ag-name">{f.name}<span class="ag-ext">.md</span></span>
              <span class="ag-sub">{f.description || `${f.source} agent`}</span>
            </span>
            <span class="ag-marks">
              {#if running > 0}<Badge variant="action" label={`${running} running`}>{running}</Badge
                >{/if}
              {#if f.readonly}<Icon name="lock" size={11} class="ag-lock" />{/if}
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="agents-ed" data-testid={TESTID.agentEditor}>
    {#if !selectedFile}
      <div class="pad">
        <PaneState kind="empty" title="Select an agent">
          {#snippet hint()}<span>Pick an agent from the list to view and edit its definition.</span
            >{/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="wk-toolbar">
        <span class="wk-crumb">agents / <b>{selectedFile.name}.md</b></span>
        <span class="ed-meta mono">{metaFor()}</span>
        {#if runningTasks > 0}
          <span class="ed-running"><Dot color="info" pulse />{runningTasks} running</span>
        {/if}
        {#if readonly}
          <span class="stag stag--mute"><Icon name="lock" size={11} />read-only</span>
        {:else if dirty}
          <span class="stag stag--warn"><Dot color="warn" />unsaved</span>
        {:else if justSaved}
          <span class="stag stag--ok"><Icon name="check" size={11} />saved</span>
        {/if}
        <div class="ed-seg" data-testid={TESTID.agentModeSeg}>
          <Seg options={['form', 'markdown']} bind:value={mode} ariaLabel="Editor mode" />
        </div>
        <div class="grow"></div>
        {#if registered}
          <Button size="sm" data-testid={TESTID.agentRun} onclick={() => (launcherOpen = true)}>
            {#snippet icon()}<Icon name="sparkle" />{/snippet}
            Run
          </Button>
        {/if}
        <Button
          size="sm"
          variant="pri"
          data-testid={TESTID.agentSave}
          disabled={readonly || !dirty}
          loading={saving}
          onclick={() => void save()}
        >
          Save
        </Button>
      </div>

      {#if loadingContent}
        <div class="pad"><PaneState kind="loading" lines={8} /></div>
      {:else if contentError}
        <div class="pad">
          <PaneState kind="error" title="Couldn't load this agent">
            {#snippet hint()}<span>{contentError}</span>{/snippet}
            {#snippet actions()}
              <Button size="sm" onclick={() => selectedPath && void loadContent(selectedPath)}
                >Retry</Button
              >
            {/snippet}
          </PaneState>
        </div>
      {:else if mode === 'form'}
        <div class="ed-pane">
          <AgentForm {summary} body={parsed.body} />
        </div>
      {:else}
        <div class="ed-pane ed-md">
          {#if readonly}
            <div class="md-note">
              <Icon name="lock" size={12} />
              This is a {sourceLabel} agent and is read-only. Copy it into your workspace to customise
              it.
            </div>
          {/if}
          <textarea
            class="mono agent-src"
            spellcheck="false"
            aria-label="Agent definition source"
            bind:value={content}
            {readonly}
            onkeydown={onEditorKeydown}></textarea>
        </div>
      {/if}
    {/if}
  </div>

  {#if selectedFile}
    <RunLauncher
      open={launcherOpen}
      {agentName}
      agentDescription={summary.description ?? selectedFile.description}
      defaultEffort={summary.effort}
      onLaunch={launch}
      onClose={() => (launcherOpen = false)}
    />
  {/if}
</section>

<style>
  /* Layout. The .t-* controls come from the component library; these classes are
     view-local. */
  .agents-shell {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    height: 100%;
    overflow: hidden;
  }
  .agents-list {
    border-right: 1px solid var(--bd0);
    background: var(--bg1);
    min-height: 0;
    display: flex;
    flex-direction: column;
  }
  .sidebar-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px 6px;
    flex: none;
  }
  .hd-title {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
  }
  .sidebar-hd .cnt {
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .sidebar-search {
    padding: 0 12px 8px;
    flex: none;
  }
  .roster {
    overflow-y: auto;
    min-height: 0;
    flex: 1;
  }
  .ag-row {
    display: flex;
    align-items: center;
    gap: 9px;
    width: 100%;
    text-align: left;
    padding: 8px 12px;
    background: none;
    border: 0;
    border-bottom: 1px solid var(--bd0);
    cursor: pointer;
    color: var(--tx1);
  }
  .ag-row:hover {
    background: var(--bg2);
  }
  .ag-row.is-selected {
    background: color-mix(in oklab, var(--acc) 12%, var(--bg1));
    box-shadow: inset 2px 0 0 var(--acc);
  }
  .ag-dot {
    flex: none;
    display: inline-flex;
  }
  .ag-main {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
  }
  .ag-name {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ag-ext {
    color: var(--tx3);
    font-weight: 400;
  }
  .ag-sub {
    font-size: var(--fs-xs);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .ag-marks {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: none;
  }
  .ag-marks :global(.ag-lock) {
    color: var(--tx3);
  }

  .agents-ed {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  .wk-toolbar {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-bottom: 1px solid var(--bd0);
    flex-wrap: wrap;
    flex: none;
  }
  .wk-crumb {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 200px;
  }
  .wk-crumb b {
    color: var(--tx1);
    font-weight: 600;
  }
  .ed-meta {
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .ed-running {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--st-info);
  }
  .stag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font: 600 var(--fs-2xs) var(--font-mono);
    padding: 2px 8px;
    border-radius: var(--r-full);
    border: 1px solid var(--bd1);
  }
  .stag--mute {
    color: var(--tx3);
    background: var(--bg1);
  }
  .stag--warn {
    color: var(--st-warn);
    border-color: color-mix(in oklab, var(--st-warn) 40%, transparent);
    background: color-mix(in oklab, var(--st-warn) 10%, transparent);
  }
  .stag--ok {
    color: var(--st-ok);
    border-color: color-mix(in oklab, var(--st-ok) 40%, transparent);
    background: color-mix(in oklab, var(--st-ok) 10%, transparent);
  }
  .stag :global(.ic) {
    flex: none;
  }
  .ed-seg {
    margin-left: 6px;
  }
  .grow {
    flex: 1;
  }
  .ed-pane {
    flex: 1;
    min-height: 0;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .ed-md {
    padding: 10px 12px 12px;
    gap: 8px;
  }
  .md-note {
    display: flex;
    align-items: center;
    gap: 7px;
    font: 400 var(--fs-xs) var(--font-ui);
    color: var(--tx2);
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 7px 10px;
    flex: none;
  }
  .md-note :global(.ic) {
    color: var(--tx3);
    flex: none;
  }
  /* Themed editor surface (no global .t-input rule exists, so the look is
     carried locally). */
  .agent-src {
    flex: 1;
    min-height: 0;
    resize: none;
    width: 100%;
    font: 400 var(--fs-xs) / 1.7 var(--font-mono);
    padding: 10px 12px;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    color: var(--tx0);
  }
  .agent-src:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .agent-src[readonly] {
    color: var(--tx2);
  }
  .pad {
    padding: 14px;
  }
  .mono {
    font-family: var(--font-mono);
  }

  @media (max-width: 640px) {
    .agents-shell {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr);
    }
    .agents-list {
      max-height: 210px;
      border-right: 0;
      border-bottom: 1px solid var(--bd0);
    }
  }
</style>
