<script lang="ts">
  // File surface: one workspace document (rendered / raw / edit) plus its metadata
  // pane (backlinks, tags, outline, related), docked as a mux tab. The file tree,
  // search, and new-note live in the shared context rail (FilesRail); this surface
  // shows whichever note it's pointed at by `params.path` and navigates internally
  // on wikilink / breadcrumb clicks. Backlinks/tags read from the shared workspace
  // index (filesWorkspace) so the whole-workspace walk is not repeated per tab.
  import { onMount, tick, untrack } from 'svelte';
  import { TESTID } from '$lib/testids';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Seg from '$lib/components/inputs/Seg.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import PhoneBack from '$lib/shell/PhoneBack.svelte';
  import { goBackToWorkspaceList } from '$lib/shell/phoneNav';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Backlinks from '$lib/components/artifact/Backlinks.svelte';
  import TagPill from '$lib/components/artifact/TagPill.svelte';
  import AnnPopover from '$lib/components/artifact/AnnPopover.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { files } from '$lib/stores/files.svelte';
  import { writeTargetsDoc } from '$lib/stores/fileWrites';
  import { agentsMeta } from '$lib/stores/agentsMeta.svelte';
  import { isMarkdown } from './load';
  import { filesWorkspace } from './workspace.svelte';
  import {
    renderMarkdown,
    stripTagsLine,
    parseHeadings,
    parseTags,
    relatedNotes,
    breadcrumbs,
    formatSize,
    formatMtime,
  } from './wiki';

  interface OpenDoc {
    path: string;
    content: string;
    markdown: boolean;
  }

  let { params }: { params?: Record<string, string> } = $props();

  const agent = $derived(params?.agent ?? agentsMeta.agents[0]?.name ?? '');

  // svelte-ignore state_referenced_locally -- seeds from the initial param; user navigation drives it after.
  let activePath = $state(params?.path ?? '');
  let browseDir = $state('');
  let doc = $state<OpenDoc | null>(null);
  let mode = $state<string>('rendered');
  let editBuffer = $state('');
  let editArea = $state<HTMLTextAreaElement | null>(null);
  let saving = $state(false);
  let staleOnDisk = $state(false);

  let docEl = $state<HTMLElement | null>(null);
  let pop = $state<{ open: boolean; x: number; y: number; text: string }>({
    open: false,
    x: 0,
    y: 0,
    text: '',
  });

  const ws = $derived(filesWorkspace.ws);
  const idx = $derived(ws?.index ?? null);
  // Backlinks/related need the content-derived index, which is built on demand
  // (a whole-workspace read) - never as a side effect of just opening a file.
  const indexed = $derived(filesWorkspace.indexState === 'ready');
  const dirty = $derived(doc != null && editBuffer !== doc.content);
  const wsName = $derived(ws?.workspaceDir.split('/').filter(Boolean).pop() ?? 'workspace');

  const renderedHtml = $derived(
    doc && doc.markdown && idx
      ? renderMarkdown(stripTagsLine(doc.content), idx.resolve)
      : doc
        ? renderMarkdown(doc.content, () => null)
        : '',
  );
  // Tags of the open doc parse from its own content, so they show without the
  // whole-workspace index; counts upgrade once a scan has run.
  const docTags = $derived(doc?.markdown ? parseTags(doc.content) : []);

  const docHtml = $derived.by(() => {
    const html = renderedHtml;
    if (docTags.length === 0) return html;
    const esc = (s: string) => s.replace(/[&<>"]/g, (c) => `&#${c.charCodeAt(0)};`);
    const chips = docTags
      .map(
        (t) =>
          `<a class="t-chip doc-tag" data-wk-tag="${esc(t)}" role="button" tabindex="0">#${esc(t)}</a>`,
      )
      .join(' ');
    const row = `<p class="doc-tags">${chips}</p>`;
    const close = html.indexOf('</h1>');
    return close >= 0
      ? `${html.slice(0, close + 5)}${row}${html.slice(close + 5)}`
      : `${row}${html}`;
  });
  const backlinkItems = $derived(
    activePath && idx && indexed
      ? (idx.backlinks.get(activePath) ?? []).map((b) => ({
          file: b.file,
          snippet: `“${b.snippet}”`,
        }))
      : [],
  );
  const outline = $derived(
    doc ? parseHeadings(doc.content).filter((h) => h.depth >= 2 && h.depth <= 3) : [],
  );
  const related = $derived(
    activePath && idx && indexed ? relatedNotes(activePath, idx.tagsByFile) : [],
  );
  const crumbs = $derived(breadcrumbs(activePath || browseDir));

  const dirRows = $derived.by(() => {
    if (!ws) return [];
    const prefix = browseDir ? `${browseDir}/` : '';
    return ws.entries
      .filter((e) => {
        if (!e.path.startsWith(prefix)) return false;
        const rest = e.path.slice(prefix.length);
        return rest.length > 0 && !rest.includes('/');
      })
      .sort((a, b) => Number(b.is_dir) - Number(a.is_dir) || a.name.localeCompare(b.name));
  });

  async function openFile(path: string) {
    closePopover();
    try {
      const file = await files.read(agent, path);
      const markdown = isMarkdown(path);
      doc = { path, content: file.content ?? '', markdown };
      activePath = path;
      editBuffer = file.content ?? '';
      staleOnDisk = false;
      const parts = path.split('/');
      parts.pop();
      browseDir = parts.join('/');
      mode = file.is_text === false ? 'raw' : markdown ? 'rendered' : 'raw';
    } catch (err) {
      toasts.push('err', 'Could not open file', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  function openDir(path: string) {
    closePopover();
    activePath = '';
    doc = null;
    browseDir = path;
  }

  async function save() {
    if (!doc || saving) return;
    saving = true;
    try {
      await files.write(agent, doc.path, editBuffer);
      doc = { ...doc, content: editBuffer };
      staleOnDisk = false;
      toasts.push('ok', 'Saved', { body: doc.path });
      // Contents changed -> the shared backlink/tag index may have shifted.
      await filesWorkspace.reload(agent);
    } catch (err) {
      toasts.push('err', 'Save failed', { body: err instanceof Error ? err.message : String(err) });
    } finally {
      saving = false;
    }
  }

  function discard() {
    if (doc) editBuffer = doc.content;
    mode = doc?.markdown ? 'rendered' : 'raw';
  }

  async function reloadFromDisk() {
    if (!doc) return;
    try {
      const file = await files.read(agent, doc.path);
      const content = file.content ?? '';
      doc = { ...doc, content };
      editBuffer = content;
      staleOnDisk = false;
    } catch (err) {
      toasts.push('err', 'Could not reload file', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  async function attachToChat() {
    if (!activePath) return;
    try {
      const attached = await files.attach(agent, activePath);
      const name = attached[0]?.name ?? activePath;
      toasts.push('ok', 'Attached to chat', { body: `${name} copied into uploads/` });
    } catch (err) {
      toasts.push('err', 'Attach failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  function onCrumb(index: number) {
    const isFileLeaf = index === crumbs.length - 1 && activePath !== '';
    if (isFileLeaf) return;
    openDir(crumbs.slice(0, index + 1).join('/'));
  }

  function pageName(path: string): string {
    return (path.split('/').pop() ?? path).replace(/\.md$/i, '');
  }

  function scrollToHeading(i: number) {
    docEl?.querySelectorAll<HTMLElement>('.doc-md h2, .doc-md h3')[i]?.scrollIntoView({
      block: 'start',
      behavior: 'smooth',
    });
  }

  // --- rendered-doc interactions: wikilink navigation + selection popover ---
  function wikilinkFromEvent(target: EventTarget | null): HTMLElement | null {
    if (!(target instanceof HTMLElement)) return null;
    return target.closest<HTMLElement>('[data-wk-nav],[data-wk-missing]');
  }
  function activateWikilink(el: HTMLElement) {
    const nav = el.getAttribute('data-wk-nav');
    if (nav) {
      void openFile(nav);
      return;
    }
    const missing = el.getAttribute('data-wk-missing');
    if (missing) toasts.push('info', 'No such page yet', { body: `[[${missing}]] is unwritten` });
  }
  function tagFromEvent(target: EventTarget | null): string | null {
    if (!(target instanceof HTMLElement)) return null;
    return target.closest<HTMLElement>('[data-wk-tag]')?.getAttribute('data-wk-tag') ?? null;
  }
  function onDocClick(e: MouseEvent) {
    const tag = tagFromEvent(e.target);
    if (tag) {
      e.preventDefault();
      return;
    }
    const link = wikilinkFromEvent(e.target);
    if (link) {
      e.preventDefault();
      activateWikilink(link);
    }
  }
  function onDocKeydown(e: KeyboardEvent) {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    const link = wikilinkFromEvent(e.target);
    if (link) {
      e.preventDefault();
      activateWikilink(link);
    }
  }

  function closePopover() {
    pop = { ...pop, open: false };
  }
  function onDocMouseUp() {
    if (mode !== 'rendered') {
      closePopover();
      return;
    }
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !docEl) {
      closePopover();
      return;
    }
    const text = sel.toString().trim();
    if (!text || !docEl.contains(sel.anchorNode)) {
      closePopover();
      return;
    }
    const range = sel.getRangeAt(0).getBoundingClientRect();
    const host = docEl.getBoundingClientRect();
    pop = {
      open: true,
      x: Math.max(4, range.left - host.left),
      y: range.bottom - host.top + docEl.scrollTop + 6,
      text,
    };
  }
  async function popComment() {
    const quote = pop.text
      .split('\n')
      .map((l) => `> ${l}`)
      .join('\n');
    await copy(`${quote}\n> source: [[${pageName(activePath)}]]`, 'Selection copied as a quote');
    closePopover();
  }
  async function popAsk() {
    closePopover();
    await attachToChat();
  }
  async function popCopyRef() {
    await copy(`[[${pageName(activePath)}]]`, 'Reference copied');
    closePopover();
  }
  async function copy(text: string, okMsg: string) {
    try {
      await navigator.clipboard?.writeText(text);
      toasts.push('ok', okMsg);
    } catch {
      toasts.push('warn', 'Clipboard unavailable');
    }
  }

  onMount(() => {
    void resolveAndOpen();
    const onScroll = () => closePopover();
    window.addEventListener('scroll', onScroll, true);
    return () => window.removeEventListener('scroll', onScroll, true);
  });

  async function resolveAndOpen() {
    if (agentsMeta.agents.length === 0) await agentsMeta.load();
    if (!agent) return;
    await filesWorkspace.ensure(agent);
    if (params?.path) await openFile(params.path);
  }

  // Focus the textarea when entering edit mode.
  $effect(() => {
    if (mode === 'edit') void tick().then(() => editArea?.focus());
  });

  // A rail click retargets this tab in place (spaces.openReusing rewrites the
  // tab's params); follow the pointed-at document. Internal wikilink/breadcrumb
  // navigation stays untouched (it doesn't change the param value).
  const paramPath = $derived(params?.path);
  $effect(() => {
    const path = paramPath;
    if (!path) return;
    untrack(() => {
      if (path !== activePath) void openFile(path);
    });
  });

  // Pull the new content in, unless the editor holds unsaved edits - only the
  // user may discard those, so they get the stale strip instead.
  $effect(() => {
    const write = files.lastWrite;
    if (!write) return;
    untrack(() => {
      if (!doc || !ws || !writeTargetsDoc(write.path, doc.path, ws.workspaceDir)) return;
      if (dirty) staleOnDisk = true;
      else void reloadFromDisk();
    });
  });
</script>

<section class="wk-shell">
  <section class="wk-main" aria-label="Document">
    <div class="wk-toolbar">
      <PhoneBack label="Back to files" onBack={() => goBackToWorkspaceList('files')} />
      <span class="wk-crumb" aria-label="Breadcrumb">
        <button type="button" class="crumb-seg" onclick={() => openDir('')}>{wsName}</button>
        {#each crumbs as seg, i (i)}
          <span class="sep">/</span>
          {#if i === crumbs.length - 1 && activePath}
            <b>{seg}</b>
          {:else}
            <button type="button" class="crumb-seg" onclick={() => onCrumb(i)}>{seg}</button>
          {/if}
        {/each}
      </span>
      <div class="grow"></div>
      {#if doc}
        <span data-testid={TESTID.filesModeSeg}>
          <Seg
            options={doc.markdown ? ['rendered', 'raw', 'edit'] : ['raw', 'edit']}
            bind:value={mode}
            ariaLabel="Document view"
          />
        </span>
        <button
          type="button"
          class="hd-btn"
          aria-label="Attach to chat"
          data-testid={TESTID.filesAttach}
          onclick={attachToChat}
        >
          <Icon name="link" />
        </button>
      {/if}
    </div>

    {#if staleOnDisk}
      <div class="wk-stale" role="status" data-testid={TESTID.filesStale}>
        <Icon name="alert" />
        <span>An agent changed this file. Your unsaved edits are still here.</span>
        <Button size="sm" onclick={reloadFromDisk}>Reload from disk</Button>
      </div>
    {/if}

    <!-- svelte-ignore a11y_no_static_element_interactions -->
    <div
      class="wk-doc"
      data-view={mode}
      data-testid={TESTID.filesDoc}
      bind:this={docEl}
      onmouseup={onDocMouseUp}
    >
      {#if filesWorkspace.error && !doc}
        <PaneState kind="error" title="Could not load workspace">
          {#snippet hint()}<span class="mono">{filesWorkspace.error}</span>{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={() => filesWorkspace.reload(agent)}>Retry</Button>
          {/snippet}
        </PaneState>
      {:else if filesWorkspace.loading && !doc}
        <PaneState kind="loading" lines={7} />
      {:else if doc}
        {#if mode === 'rendered'}
          <!-- svelte-ignore a11y_no_noninteractive_element_interactions -->
          <div class="doc-md" role="document" onclick={onDocClick} onkeydown={onDocKeydown}>
            {@html docHtml}
          </div>
          <AnnPopover
            open={pop.open}
            x={pop.x}
            y={pop.y}
            onComment={popComment}
            onAsk={popAsk}
            onCopyRef={popCopyRef}
          />
        {:else if mode === 'raw'}
          <pre class="rawpre">{doc.content}</pre>
        {:else}
          <div class="wk-edit">
            <textarea
              bind:this={editArea}
              bind:value={editBuffer}
              class="doc-edit"
              spellcheck="false"
              aria-label={`Edit ${doc.path}`}></textarea>
            <div class="edit-row">
              <Button variant="pri" size="sm" loading={saving} disabled={!dirty} onclick={save}>
                Save
              </Button>
              <Button variant="ghost" size="sm" disabled={!dirty} onclick={discard}>Discard</Button>
              <span class="edit-note mono">
                {dirty ? 'unsaved changes' : 'saved · agents see edits immediately'}
              </span>
            </div>
          </div>
        {/if}
      {:else}
        <div class="dir-browse" data-testid={TESTID.filesDirTable}>
          {#if dirRows.length === 0}
            <PaneState kind="empty" title="Empty directory">
              {#snippet hint()}<span>Nothing here yet. Create a note from the file rail.</span
                >{/snippet}
            </PaneState>
          {:else}
            <table class="dir-table" aria-label="Directory contents">
              <thead>
                <tr><th>name</th><th class="num">size</th><th>modified</th></tr>
              </thead>
              <tbody>
                {#each dirRows as e (e.path)}
                  <tr>
                    <td>
                      <button
                        type="button"
                        class="dir-link"
                        onclick={() => (e.is_dir ? openDir(e.path) : openFile(e.path))}
                      >
                        <Icon name={e.is_dir ? 'files' : 'file'} />{e.name}
                      </button>
                    </td>
                    <td class="num c3">{e.is_dir ? '' : formatSize(e.size ?? 0)}</td>
                    <td class="c3">{formatMtime(e.modified)}</td>
                  </tr>
                {/each}
              </tbody>
            </table>
          {/if}
        </div>
      {/if}
    </div>
  </section>

  <aside class="wk-meta" data-testid={TESTID.filesMeta} aria-label="Document metadata">
    {#if doc}
      {#if indexed}
        <div data-testid={TESTID.filesBacklinks}>
          <Backlinks
            heading={`backlinks · ${backlinkItems.length}`}
            links={backlinkItems}
            onSelect={(link) => openFile(link.file)}
          />
        </div>
      {:else}
        <div class="meta-sec">
          <h4>backlinks · related</h4>
          {#if filesWorkspace.indexState === 'building'}
            <span class="spec-note">scanning workspace…</span>
          {:else}
            <span class="spec-note">
              Backlinks and related notes need a one-time scan that reads every note in this
              workspace.
            </span>
            <div class="scan-row">
              <Button size="sm" onclick={() => void filesWorkspace.ensureIndex()}>
                Scan workspace
              </Button>
            </div>
          {/if}
        </div>
      {/if}
      {#if docTags.length > 0}
        <div class="meta-sec">
          <h4>tags</h4>
          <div class="tag-row">
            {#each docTags as tag (tag)}
              <TagPill {tag} count={idx?.tagCounts.get(tag) ?? 1} onSelect={() => {}} />
            {/each}
          </div>
        </div>
      {/if}
      {#if outline.length > 0}
        <div class="meta-sec wk-out">
          <h4>outline</h4>
          {#each outline as h, i (i)}
            <button
              type="button"
              class="out-link"
              class:sub={h.depth === 3}
              onclick={() => scrollToHeading(i)}
            >
              {h.text}
            </button>
          {/each}
        </div>
      {/if}
      {#if indexed}
        <div class="meta-sec">
          <h4>related</h4>
          {#if related.length > 0}
            <span class="spec-note"
              >{related.length} {related.length === 1 ? 'note shares' : 'notes share'} ≥2 tags</span
            >
            <div class="related-list">
              {#each related.slice(0, 6) as r (r.path)}
                <button type="button" class="related-link" onclick={() => openFile(r.path)}>
                  {pageName(r.path)}<span class="c3"> · {r.shared}</span>
                </button>
              {/each}
            </div>
          {:else}
            <span class="spec-note">No related notes.</span>
          {/if}
        </div>
      {/if}
    {:else}
      <p class="spec-note">Select a note to see its backlinks, tags, and outline.</p>
    {/if}
  </aside>
</section>

<style>
  .wk-shell {
    display: grid;
    /* The column carries its own width, so the track collapses when it hides.
       A fixed track would stay reserved and leave dead space in a narrow pane. */
    grid-template-columns: minmax(0, 1fr) auto;
    flex: 1;
    min-height: 0;
    min-width: 0;
    background: var(--bg0);
    container-type: inline-size;
  }
  .wk-main {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .hd-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 24px;
    height: 24px;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: none;
    color: var(--tx2);
    cursor: pointer;
    flex: none;
  }
  .hd-btn:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  .hd-btn:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .hd-btn :global(.ic) {
    width: 13px;
    height: 13px;
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
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .wk-crumb .crumb-seg {
    background: none;
    border: 0;
    color: var(--tx3);
    font: inherit;
    cursor: pointer;
    padding: 0;
  }
  .wk-crumb .crumb-seg:hover {
    color: var(--acc);
  }
  .wk-crumb b {
    color: var(--tx1);
    font-weight: 600;
  }
  .wk-crumb .sep {
    color: var(--tx3);
  }
  .wk-stale {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-bottom: 1px solid var(--st-warn);
    background: var(--bg2);
    color: var(--tx1);
    font-size: var(--fs-xs);
    flex: none;
    flex-wrap: wrap;
  }
  .wk-stale :global(svg) {
    color: var(--st-warn);
    flex: none;
  }
  .wk-doc {
    flex: 1;
    overflow-y: auto;
    padding: 16px 20px 30px;
    position: relative;
    min-height: 0;
  }
  .wk-doc :global(.doc-md) {
    font-size: var(--fs-md);
    line-height: 1.62;
    color: var(--tx1);
    max-width: 72ch;
  }
  .wk-doc :global(.doc-md > :first-child) {
    margin-top: 0;
  }
  .wk-doc :global(.doc-md h1) {
    font: 600 var(--fs-2xl) / 1.25 var(--font-ui);
    letter-spacing: -0.01em;
    color: var(--tx0);
    margin: 0 0 10px;
  }
  .wk-doc :global(.doc-md h2) {
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
    color: var(--tx0);
    margin: 20px 0 7px;
    padding-bottom: 4px;
    border-bottom: 1px solid var(--bd0);
  }
  .wk-doc :global(.doc-md h3) {
    font: 600 var(--fs-md) / 1.3 var(--font-ui);
    color: var(--tx0);
    margin: 16px 0 6px;
  }
  .wk-doc :global(.doc-md p) {
    margin: 7px 0;
  }
  .wk-doc :global(.doc-md ul),
  .wk-doc :global(.doc-md ol) {
    margin: 6px 0;
    padding-left: 20px;
  }
  .wk-doc :global(.doc-md li) {
    margin: 3px 0;
  }
  .wk-doc :global(.doc-md li input[type='checkbox']) {
    accent-color: var(--st-ok);
    margin: 0 6px 0 0;
    translate: 0 1px;
  }
  .wk-doc :global(.doc-md code) {
    font: 500 var(--fs-sm) var(--font-mono);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    padding: 0 4px;
    border-radius: 4px;
    color: var(--tx0);
  }
  .wk-doc :global(.doc-md pre) {
    background: var(--bg1);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    padding: 10px 12px;
    overflow-x: auto;
  }
  .wk-doc :global(.doc-md pre code) {
    background: none;
    border: 0;
    padding: 0;
  }
  .wk-doc :global(.doc-md .doc-tags) {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin: 0 0 12px;
  }
  .wk-doc :global(.doc-md .doc-tag) {
    display: inline-flex;
    align-items: center;
    height: 18px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    text-decoration: none;
    cursor: pointer;
    white-space: nowrap;
  }
  .wk-doc :global(.doc-md .doc-tag:hover),
  .wk-doc :global(.doc-md .doc-tag:focus-visible) {
    border-color: var(--acc);
    color: var(--acc);
    outline: none;
  }
  .wk-doc :global(.doc-md table) {
    border-collapse: collapse;
    margin: 10px 0;
    font-size: var(--fs-sm);
  }
  .wk-doc :global(.doc-md th) {
    text-align: left;
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--tx3);
    border-bottom: 1px solid var(--bd1);
    padding: 5px 12px 5px 0;
  }
  .wk-doc :global(.doc-md td) {
    border-bottom: 1px solid var(--bd0);
    padding: 5px 12px 5px 0;
    color: var(--tx1);
  }
  .wk-doc :global(.doc-md blockquote) {
    border-left: 3px solid color-mix(in oklab, var(--st-info) 45%, transparent);
    background: color-mix(in oklab, var(--st-info) 8%, transparent);
    border-radius: var(--r-md);
    padding: 8px 12px;
    margin: 10px 0;
    color: var(--tx1);
  }
  /* Frontmatter kv panel: long scalar values wrap instead of stretching the
     table past the doc column, and nested blocks wrap inside their cell. */
  .wk-doc :global(.doc-md .tsu-fm table) {
    table-layout: fixed;
    width: 100%;
  }
  .wk-doc :global(.doc-md .tsu-fm th) {
    width: 14ch;
    vertical-align: top;
  }
  .wk-doc :global(.doc-md .tsu-fm td) {
    overflow-wrap: anywhere;
  }
  .wk-doc :global(.doc-md .tsu-fm td pre) {
    margin: 0;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .wk-doc :global(.doc-md blockquote p) {
    margin: 0;
  }
  .wk-doc :global(.doc-md strong) {
    color: var(--tx0);
    font-weight: 600;
  }
  .wk-doc :global(.doc-md a:not(.wikilink)) {
    color: var(--acc);
  }
  .wk-doc :global(.doc-md .wikilink) {
    color: var(--brand);
    border-bottom: 1px dashed color-mix(in oklab, var(--brand) 55%, transparent);
    cursor: pointer;
    text-decoration: none;
  }
  .wk-doc :global(.doc-md .wikilink:hover),
  .wk-doc :global(.doc-md .wikilink:focus-visible) {
    color: color-mix(in oklab, var(--brand) 75%, var(--tx0));
    border-bottom-style: solid;
    outline: none;
  }
  .wk-doc :global(.doc-md .wikilink.is-missing) {
    color: var(--st-err);
    border-bottom-color: color-mix(in oklab, var(--st-err) 55%, transparent);
  }
  .wk-doc :global(.vh) {
    position: absolute;
    width: 1px;
    height: 1px;
    margin: -1px;
    padding: 0;
    overflow: hidden;
    clip: rect(0 0 0 0);
    white-space: nowrap;
    border: 0;
  }
  .rawpre {
    margin: 0;
    padding: 12px 14px;
    font: 400 var(--fs-xs) / 1.7 var(--font-mono);
    color: var(--tx2);
    white-space: pre-wrap;
    word-break: break-word;
  }
  .wk-edit {
    display: grid;
    gap: 8px;
  }
  .doc-edit {
    width: 100%;
    min-height: 340px;
    resize: vertical;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 12px 14px;
    color: var(--tx0);
    font: 400 var(--fs-sm) / 1.7 var(--font-mono);
  }
  .doc-edit:focus {
    outline: none;
    border-color: var(--acc);
    box-shadow: 0 0 0 3px color-mix(in oklab, var(--acc) 22%, transparent);
  }
  .edit-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .edit-note {
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .dir-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--fs-sm);
  }
  .dir-table th {
    text-align: left;
    font: 600 var(--fs-2xs) var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
    padding: 6px 10px;
    border-bottom: 1px solid var(--bd1);
  }
  .dir-table th.num,
  .dir-table td.num {
    text-align: right;
    font-variant-numeric: tabular-nums;
  }
  .dir-table td {
    padding: 4px 10px;
    border-bottom: 1px solid var(--bd0);
  }
  .dir-table td.c3 {
    color: var(--tx3);
    font: 400 var(--fs-xs) var(--font-mono);
    white-space: nowrap;
  }
  .dir-link {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: none;
    border: 0;
    color: var(--tx1);
    font: 500 var(--fs-sm) var(--font-mono);
    cursor: pointer;
    padding: 0;
  }
  .dir-link:hover {
    color: var(--acc);
  }
  .dir-link :global(.ic) {
    width: 12px;
    height: 12px;
    color: var(--tx3);
    flex: none;
  }
  .wk-meta {
    display: flex;
    flex-direction: column;
    width: 208px;
    border-left: 1px solid var(--bd0);
    background: var(--bg1);
    overflow-y: auto;
    padding: 10px 12px;
    gap: 14px;
    min-height: 0;
  }
  .meta-sec h4 {
    margin: 0 0 6px;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .scan-row {
    margin-top: 7px;
  }
  .tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
  }
  .wk-out .out-link {
    display: block;
    width: 100%;
    text-align: left;
    background: none;
    border: 0;
    font-size: var(--fs-xs);
    color: var(--tx2);
    padding: 2px 0;
    cursor: pointer;
  }
  .wk-out .out-link.sub {
    padding-left: 10px;
    color: var(--tx3);
  }
  .wk-out .out-link:hover {
    color: var(--acc);
  }
  .spec-note {
    font: 400 var(--fs-xs) / 1.5 var(--font-mono);
    color: var(--tx3);
  }
  .related-list {
    display: grid;
    gap: 3px;
    margin-top: 5px;
  }
  .related-link {
    text-align: left;
    background: none;
    border: 0;
    color: var(--tx2);
    font: 500 var(--fs-xs) var(--font-mono);
    cursor: pointer;
    padding: 2px 0;
  }
  .related-link:hover {
    color: var(--acc);
  }
  .c3 {
    color: var(--tx3);
  }
  .mono {
    font-family: var(--font-mono);
  }
  @container (max-width: 720px) {
    .wk-meta {
      display: none;
    }
  }
</style>
