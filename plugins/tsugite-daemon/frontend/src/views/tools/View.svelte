<script lang="ts">
  // Tool registry browser: GET /api/tools over the same read-only, no-mutation
  // surface as the plugins registry. A daemon predating that endpoint answers
  // 404, which the store degrades to `available:false` - rendered here as a
  // truthful "not exposed" state, not an error.
  import { tools } from '$lib/stores/tools.svelte';
  import { filterTools } from './filterTools';
  import Chip from '$lib/components/buttons/Chip.svelte';
  import SearchInput from '$lib/components/inputs/SearchInput.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Table, { type TableColumn } from '$lib/components/datadisplay/Table.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { TESTID } from '$lib/testids';

  let query = $state('');

  $effect(() => {
    tools.load();
  });

  const filtered = $derived(filterTools(tools.tools, query));
  const columns: TableColumn[] = [
    { key: 'tool', label: 'tool' },
    { key: 'category', label: 'category' },
    { key: 'description', label: 'description' },
    { key: 'source', label: 'source' },
  ];
</script>

<!-- No aria-label here: the mux Pane already landmarks this region via its tab
     title, so a second same-named region would just duplicate the landmark. -->
<section class="tools-view" data-testid={TESTID.view('tools')}>
  <div class="head">
    <h2>Tools</h2>
    {#if tools.available && tools.tools.length > 0}
      <span class="count">{tools.tools.length} {tools.tools.length === 1 ? 'tool' : 'tools'}</span>
    {/if}
    <div class="grow"></div>
    <SearchInput
      bind:value={query}
      placeholder="search tools…"
      ariaLabel="Search tools"
      disabled={!tools.available || tools.tools.length === 0}
    />
  </div>
  <p class="note">
    Builtin tools ship with tsugite; plugin tools are registered by installed plugins.
  </p>

  {#if tools.loading && tools.tools.length === 0}
    <PaneState kind="loading" />
  {:else if tools.error}
    <PaneState kind="error" title="Couldn't load tools">
      {#snippet icon()}<Icon name="alert" />{/snippet}
      {#snippet hint()}<span class="mono">{tools.error}</span>{/snippet}
      {#snippet actions()}
        <Button size="sm" onclick={() => tools.load()}>
          {#snippet icon()}<Icon name="retry" />{/snippet}
          Retry
        </Button>
      {/snippet}
    </PaneState>
  {:else if !tools.available || tools.tools.length === 0}
    <PaneState kind="empty" title="No tools registered">
      {#snippet icon()}<Icon name="tool" />{/snippet}
      {#snippet hint()}This daemon has no builtin or plugin-registered tools.{/snippet}
    </PaneState>
  {:else if filtered.length === 0}
    <PaneState kind="empty" title="No matching tools">
      {#snippet icon()}<Icon name="search" />{/snippet}
      {#snippet hint()}No tool matches <span class="mono">"{query.trim()}"</span>.{/snippet}
      {#snippet actions()}
        <Button size="sm" onclick={() => (query = '')}>Clear search</Button>
      {/snippet}
    </PaneState>
  {:else}
    <div class="tbl-wrap">
      <Table {columns} rows={filtered} key={(tool) => tool.name} ariaLabel="Tools">
        {#snippet cells(tool)}
          <td class="mono">{tool.name}</td>
          <td><Chip>{tool.category ?? '—'}</Chip></td>
          <td class="c2 desc" title={tool.description || undefined}>
            {#if tool.description}
              {tool.description}
            {:else}
              <span class="c3">no description</span>
            {/if}
          </td>
          <td class="c2">
            {#if tool.source === 'plugin'}
              <span class="src"><Icon name="plug" size={11} />plugin</span>
            {:else}
              {tool.source ?? 'builtin'}
            {/if}
          </td>
        {/snippet}
      </Table>
    </div>
  {/if}
</section>

<style>
  /* Fills the mux pane body and scrolls internally (rather than letting
     `.mux-bd` scroll the whole view) so the head/search stay put while the
     table scrolls - and so the table's own sticky header has an actual
     scrolling ancestor to stick within. `.tbl-wrap` needing `overflow-x` for
     wide viewports otherwise computes its `overflow-y` to auto too (CSS
     Overflow §3), making it the sticky containing block; without a bounded
     height of its own that block never scrolls, so `position: sticky` goes
     inert - confirmed empirically, not just by spec-reading. */
  .tools-view {
    height: 100%;
    box-sizing: border-box;
    padding: 14px 16px 26px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    min-height: 0;
  }
  .head {
    flex: none;
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .head h2 {
    margin: 0;
    font: 600 var(--fs-lg) var(--font-ui);
  }
  .count {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .head :global(.t-search) {
    width: 220px;
    max-width: 100%;
  }
  .note {
    flex: none;
    margin: 0;
    font: 400 var(--fs-xs) / 1.5 var(--font-ui);
    color: var(--tx3);
  }

  .tbl-wrap {
    flex: 1;
    min-height: 0;
    overflow: auto;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
  }
  .desc {
    max-width: 46ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .src {
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
</style>
