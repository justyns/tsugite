<script lang="ts">
  // Skills catalog: discovered SKILL.md manifests (builtin/project/global) with
  // scan + load diagnostics, over GET /api/skill-files + GET /api/skills/issues.
  // Laid out like the Agents builder: a roster of skills on the left and a
  // full-height detail pane on the right (overview, diagnostics, and the raw
  // SKILL.md source, loaded on select).
  //
  // No per-skill enable/disable exists anywhere in the backend - no config
  // field, no HTTP mutation (grepped skill_discovery.py, tools/skills.py,
  // config.py). Same call already made independently elsewhere in this
  // rebuild for plugins/webhooks/schedules: no toggle for a field the daemon
  // has no way to persist. Rendered as a read-only catalog instead.
  //
  // The two endpoints also disagree on skill roots: skill-files is scoped per
  // configured agent's workspace_dir; skills/issues comes from a
  // process-lifetime singleton that scans CWD-relative roots (+
  // ~/.agents/skills) once at first access and never rescans - unrelated to
  // any agent's actual workspace_dir. buildSkillCatalog() (skillCatalog.ts)
  // joins the two defensively by path/name and never drops an issue that
  // fails to join - an unmatched issue still surfaces as its own row instead
  // of silently vanishing.
  import { untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { nextRovingIndex } from '$lib/actions/rovingNav';
  import {
    buildSkillCatalog,
    catalogHeading,
    catalogSummary,
    filterSkillCatalog,
    issuesHeading,
    skillStatusLabel,
    type SkillCatalogRow,
  } from './skillCatalog';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Dot from '$lib/components/buttons/Dot.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  let query = $state('');
  let loading = $state(true);
  let error = $state<string | null>(null);
  let selectedPath = $state<string | null>(null);
  let sourceCache = $state<Record<string, string>>({});
  let sourceLoading = $state(false);
  let sourceError = $state<string | null>(null);

  async function loadCatalog(): Promise<void> {
    loading = true;
    error = null;
    try {
      await Promise.all([agentsMeta.loadSkillFiles(), agentsMeta.loadSkillIssues()]);
    } catch (err) {
      error = err instanceof Error ? err.message : String(err);
    } finally {
      loading = false;
    }
  }

  $effect(() => {
    loadCatalog();
  });

  const catalog = $derived(buildSkillCatalog(agentsMeta.skillFiles, agentsMeta.skillIssues));
  const filtered = $derived(filterSkillCatalog(catalog, query));
  const summary = $derived(catalogSummary(catalog));
  const selected = $derived(catalog.find((row) => row.path === selectedPath) ?? null);
  const selectedVisible = $derived(filtered.some((row) => row.path === selectedPath));

  // Auto-select the first skill once the catalog lands, like the agents roster.
  $effect(() => {
    if (!selectedPath && catalog.length > 0) {
      const first = catalog[0]!.path;
      untrack(() => selectRow(first));
    }
  });

  function selectRow(path: string): void {
    selectedPath = path;
    sourceError = null;
    const row = catalog.find((r) => r.path === path);
    if (row && !row.synthetic && !(path in sourceCache)) void fetchSource(row);
  }

  async function fetchSource(row: SkillCatalogRow): Promise<void> {
    sourceLoading = true;
    sourceError = null;
    try {
      const res = await agentsMeta.readSkillFile(row.path);
      sourceCache[row.path] = res.content;
    } catch (err) {
      sourceError = err instanceof Error ? err.message : String(err);
    } finally {
      sourceLoading = false;
    }
  }

  function onRowKeydown(e: KeyboardEvent, index: number): void {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      const row = filtered[index];
      if (row) selectRow(row.path);
      return;
    }
    const next = nextRovingIndex(index, e.key, filtered.length);
    if (next === null) return;
    e.preventDefault();
    const host = e.currentTarget as HTMLElement;
    const rows = host.closest('[role="listbox"]')?.querySelectorAll<HTMLElement>('[data-rowidx]');
    rows?.[next]?.focus();
  }
</script>

{#snippet statusPill(row: SkillCatalogRow)}
  <span class="t-pill" data-st={row.status}>
    {#if row.status === 'ok'}
      <Icon name="check" size={11} />
    {:else if row.status === 'warning'}
      <Icon name="alert" size={11} />
    {:else}
      <Icon name="x" size={11} />
    {/if}
    {skillStatusLabel(row)}
  </span>
{/snippet}

<section class="skl-shell" data-testid={TESTID.view('skills')}>
  <div class="skl-list">
    <div class="sidebar-hd">
      <strong class="hd-title">Skills</strong>
      {#if !loading && !error}
        <span class="count mono">{catalogHeading(summary)}</span>
      {/if}
    </div>
    <div class="sidebar-search">
      <SearchInput
        bind:value={query}
        placeholder="filter skills…"
        ariaLabel="Filter skills"
        shortcutKey="/"
        disabled={catalog.length === 0}
      />
    </div>

    {#if loading && catalog.length === 0}
      <div class="pad"><PaneState kind="loading" lines={6} /></div>
    {:else if error && catalog.length === 0}
      <div class="pad">
        <PaneState kind="error" title="Couldn't load skills">
          {#snippet icon()}<Icon name="alert" />{/snippet}
          {#snippet hint()}<span class="mono">{error}</span>{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={loadCatalog}>
              {#snippet icon()}<Icon name="retry" />{/snippet}
              Retry
            </Button>
          {/snippet}
        </PaneState>
      </div>
    {:else if catalog.length === 0}
      <div class="pad">
        <PaneState kind="empty" title="No skills discovered">
          {#snippet icon()}<Icon name="skill" />{/snippet}
          {#snippet hint()}Drop a <span class="mono">SKILL.md</span> under a workspace's
            <span class="mono">skills/</span> directory to add one.{/snippet}
        </PaneState>
      </div>
    {:else if filtered.length === 0}
      <div class="pad">
        <PaneState kind="empty" title="No skills match your filter">
          {#snippet icon()}<Icon name="search" />{/snippet}
          {#snippet hint()}Try a different name, description, or source.{/snippet}
          {#snippet actions()}
            <Button size="sm" variant="ghost" onclick={() => (query = '')}>Clear filter</Button>
          {/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="roster" role="listbox" aria-label="Skills" tabindex="-1">
        {#each filtered as row, i (row.path)}
          {@const isSel = row.path === selectedPath}
          <button
            type="button"
            class="skl-row"
            class:is-selected={isSel}
            role="option"
            aria-selected={isSel}
            tabindex={isSel || (!selectedVisible && i === 0) ? 0 : -1}
            data-rowidx={i}
            onclick={() => selectRow(row.path)}
            onkeydown={(e) => onRowKeydown(e, i)}
          >
            <span class="skl-dot">
              <Dot color={row.status === 'ok' ? 'ok' : row.status === 'warning' ? 'warn' : 'err'} />
            </span>
            <span class="skl-main">
              <span class="skl-name">{row.name}</span>
              <span class="skl-sub">{row.description || skillStatusLabel(row)}</span>
            </span>
            <span class="skl-marks">
              {#if row.issues.length > 0}
                <span class="issue-n" data-sev={row.status}>{row.issues.length}</span>
              {/if}
              {#if row.readonly}<Icon name="lock" size={11} class="skl-lock" />{/if}
            </span>
          </button>
        {/each}
      </div>
    {/if}
  </div>

  <div class="skl-detail" data-testid={TESTID.skillsDrawer}>
    {#if !selected}
      <div class="pad">
        <PaneState kind="empty" title="Select a skill">
          {#snippet hint()}<span
              >Pick a skill from the list to see its manifest and diagnostics.</span
            >{/snippet}
        </PaneState>
      </div>
    {:else}
      <div class="wk-toolbar">
        <span class="wk-crumb">skills / <b>{selected.name}</b></span>
        {#if selected.source}
          <span class="t-chip">
            {#if selected.readonly}<Icon name="lock" size={10} />{/if}
            {selected.source}
          </span>
        {/if}
        {@render statusPill(selected)}
        <div class="grow"></div>
      </div>

      <div class="skl-body">
        {#if selected.description}
          <p class="d-desc">{selected.description}</p>
        {/if}
        <div class="d-path mono">{selected.path}</div>

        <div class="d-sec">
          <h4>{issuesHeading(selected.issues.length)}</h4>
          {#if selected.issues.length === 0}
            <div class="d-issue" data-sev="ok"><Icon name="check" size={11} />no known issues</div>
          {:else}
            <ul class="d-issues">
              {#each selected.issues as issue, i (i)}
                <li class="d-issue" data-sev={issue.severity}>
                  <Icon name={issue.severity === 'error' ? 'x' : 'alert'} size={11} />
                  <span class="d-issue-msg">{issue.message}</span>
                  <span class="d-issue-src mono">{issue.source}</span>
                </li>
              {/each}
            </ul>
          {/if}
        </div>

        <div class="d-sec skl-srcsec">
          <h4>SKILL.md source</h4>
          {#if selected.synthetic}
            <p class="d-hint">
              This path isn't inside any configured skill directory the file browser can read - only
              the diagnostic issue above is available for it.
            </p>
          {:else if sourceLoading && !(selected.path in sourceCache)}
            <PaneState kind="loading" lines={5} />
          {:else if sourceError && !(selected.path in sourceCache)}
            <PaneState kind="error" title="Couldn't read file">
              {#snippet hint()}<span class="mono">{sourceError}</span>{/snippet}
              {#snippet actions()}
                <Button size="sm" onclick={() => selected && fetchSource(selected)}>
                  {#snippet icon()}<Icon name="retry" />{/snippet}
                  Retry
                </Button>
              {/snippet}
            </PaneState>
          {:else}
            <pre class="d-src">{sourceCache[selected.path]}</pre>
          {/if}
        </div>
      </div>
    {/if}
  </div>
</section>

<style>
  .mono {
    font-family: var(--font-mono);
  }

  /* Master-detail layout mirroring the agents builder (.agents-shell). */
  .skl-shell {
    flex: 1;
    min-height: 0;
    display: grid;
    grid-template-columns: 300px minmax(0, 1fr);
    height: 100%;
    overflow: hidden;
  }
  .skl-list {
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
  .count {
    font-size: var(--fs-2xs);
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
  .skl-row {
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
  .skl-row:hover {
    background: var(--bg2);
  }
  .skl-row.is-selected {
    background: color-mix(in oklab, var(--acc) 12%, var(--bg1));
    box-shadow: inset 2px 0 0 var(--acc);
  }
  .skl-dot {
    flex: none;
    display: inline-flex;
  }
  .skl-main {
    display: flex;
    flex-direction: column;
    gap: 1px;
    min-width: 0;
    flex: 1;
  }
  .skl-name {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .skl-sub {
    font-size: var(--fs-xs);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .skl-marks {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    flex: none;
  }
  .skl-marks :global(.skl-lock) {
    color: var(--tx3);
  }
  .issue-n {
    display: inline-grid;
    place-items: center;
    min-width: 17px;
    height: 16px;
    padding: 0 5px;
    border-radius: var(--r-sm);
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx2);
  }
  .issue-n[data-sev='error'] {
    color: var(--st-err);
    border-color: color-mix(in oklab, var(--st-err) 40%, transparent);
  }
  .issue-n[data-sev='warning'] {
    color: var(--st-warn);
    border-color: color-mix(in oklab, var(--st-warn) 40%, transparent);
  }

  /* ---- detail pane ---- */
  .skl-detail {
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
    max-width: 260px;
  }
  .wk-crumb b {
    color: var(--tx1);
    font-weight: 600;
  }
  .grow {
    flex: 1;
  }
  .skl-body {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: 12px 14px 18px;
    display: flex;
    flex-direction: column;
    gap: 14px;
  }
  .d-desc {
    margin: 0;
    font: 400 var(--fs-sm) / 1.5 var(--font-ui);
    color: var(--tx1);
  }
  .d-path {
    font-size: var(--fs-xs);
    color: var(--tx3);
    word-break: break-all;
  }
  .d-hint {
    margin: 0;
    font: 400 var(--fs-xs) / 1.5 var(--font-ui);
    color: var(--tx3);
  }
  .d-sec {
    display: grid;
    gap: 6px;
  }
  .d-sec > h4 {
    margin: 0;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .d-issues {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 6px;
  }
  .d-issue {
    display: flex;
    align-items: baseline;
    gap: 7px;
    font-size: var(--fs-sm);
    color: var(--tx2);
  }
  .d-issue :global(.ic) {
    flex: none;
    align-self: center;
  }
  .d-issue-msg {
    flex: 1;
    min-width: 0;
  }
  .d-issue-src {
    flex: none;
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .d-issue[data-sev='error'] {
    color: var(--st-err);
  }
  .d-issue[data-sev='error'] :global(.ic) {
    color: var(--st-err);
  }
  .d-issue[data-sev='warning'] :global(.ic) {
    color: var(--st-warn);
  }
  .d-issue[data-sev='ok'] :global(.ic) {
    color: var(--st-ok);
  }
  /* The source section takes the remaining height so the manifest reads in a
     big pane, not a drawer-sized peephole. */
  .skl-srcsec {
    flex: 1;
    min-height: 0;
    grid-template-rows: auto minmax(0, 1fr);
  }
  .d-src {
    margin: 0;
    min-height: 200px;
    overflow: auto;
    background: var(--bg0);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 10px 12px;
    font: 400 var(--fs-xs) / 1.6 var(--font-mono);
    color: var(--tx1);
    white-space: pre-wrap;
    word-break: break-word;
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
    color: var(--tx3);
  }

  /* ---- status pill - own local data-st vocabulary (ok/warning/error), same
     precedent as mcp/Entity.svelte's popover pill: this vocabulary isn't the
     shared buttons/Pill's session-state set, so that component can't model
     it - each consumer with its own status words keeps its own copy. ---- */
  .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .t-pill[data-st='ok'] {
    --c: var(--st-ok);
  }
  .t-pill[data-st='warning'] {
    --c: var(--st-warn);
  }
  .t-pill[data-st='error'] {
    --c: var(--st-err);
  }

  .pad {
    padding: 14px;
  }

  @media (max-width: 640px) {
    .skl-shell {
      grid-template-columns: minmax(0, 1fr);
      grid-template-rows: auto minmax(0, 1fr);
    }
    .skl-list {
      max-height: 210px;
      border-right: 0;
      border-bottom: 1px solid var(--bd0);
    }
  }
</style>
