<script lang="ts">
  // Plugin registry: a read-only mirror of `tsu plugins list` (GET /api/plugins
  // over tsugite.plugins.discover_plugins()). enabled reflects daemon.yaml's
  // per-plugin config; loaded/error are real fields on the wire contract but
  // discover_plugins() is a pure entry-point scan - it never imports anything,
  // so today every row reads loaded:false, error:null regardless of what
  // actually happened at daemon boot. Rendered as-is: no toggle (enable/disable
  // has no HTTP mutation to call), no invented load/error state.
  import { TESTID } from '$lib/testids';
  import { pluginsMeta } from '$lib/stores/pluginsMeta.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import Dot from '$lib/components/buttons/Dot.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { groupLabel, pluginKey, sortPlugins } from './format';

  // Refetches on open even though the shell loads this at boot: this view owns
  // the loading / error / retry surface for the registry.
  $effect(() => {
    pluginsMeta.load();
  });

  const rows = $derived(sortPlugins(pluginsMeta.plugins));
  const enabledCount = $derived(rows.filter((p) => p.enabled).length);
  const errored = $derived(rows.filter((p) => p.error));
  const showSkeleton = $derived(pluginsMeta.loading && rows.length === 0 && !pluginsMeta.error);
</script>

<section class="plugins-view" data-testid={TESTID.view('plugins')} aria-labelledby="plugins-h">
  <div class="view-pad">
    <div class="row">
      <h2 id="plugins-h">Plugins</h2>
      {#if pluginsMeta.available && rows.length > 0}
        <span class="dim mono">{rows.length} · {enabledCount} enabled</span>
      {/if}
      <div class="grow"></div>
    </div>

    {#if showSkeleton}
      <PaneState kind="loading" />
    {:else if pluginsMeta.error}
      <PaneState kind="error" title="Couldn't load plugins">
        {#snippet icon()}<Icon name="alert" />{/snippet}
        {#snippet hint()}<span class="mono">{pluginsMeta.error}</span>{/snippet}
        {#snippet actions()}
          <Button size="sm" data-testid={TESTID.pluginsRetry} onclick={() => pluginsMeta.load()}>
            {#snippet icon()}<Icon name="retry" />{/snippet}Retry
          </Button>
        {/snippet}
      </PaneState>
    {:else if !pluginsMeta.available || rows.length === 0}
      <PaneState kind="empty" title="No plugins installed">
        {#snippet icon()}<Icon name="plug" />{/snippet}
        {#snippet hint()}Install a package with a <span class="mono">tsugite.*</span> entry point to extend
          this daemon.{/snippet}
      </PaneState>
    {:else}
      <div class="plg-tbl-wrap">
        <table class="t-table" data-testid={TESTID.pluginsTable} aria-label="Installed plugins">
          <thead>
            <tr>
              <th scope="col">name</th>
              <th scope="col">group</th>
              <th scope="col">enabled</th>
              <th scope="col">loaded</th>
            </tr>
          </thead>
          <tbody>
            {#each rows as p (pluginKey(p))}
              <tr class:is-off={!p.enabled} data-testid={TESTID.pluginRow(pluginKey(p))}>
                <td class="nm mono">{p.name}</td>
                <td><span class="t-chip mono">{groupLabel(p.group)}</span></td>
                <td>
                  <span class="stcell">
                    <Dot color={p.enabled ? 'ok' : 'mute'} />{p.enabled ? 'enabled' : 'disabled'}
                  </span>
                </td>
                <td>
                  <span class="stcell">
                    <Dot color={p.loaded ? 'ok' : 'mute'} />{p.loaded ? 'loaded' : 'not loaded'}
                  </span>
                </td>
              </tr>
            {/each}
          </tbody>
        </table>
      </div>

      {#if errored.length > 0}
        <div class="plg-errs">
          {#each errored as p (pluginKey(p))}
            <div class="t-callout t-callout--err" data-testid={TESTID.pluginError(pluginKey(p))}>
              <Icon name="alert" />
              <div><b>{p.name}</b> failed to load - {p.error}</div>
            </div>
          {/each}
        </div>
      {/if}
    {/if}
  </div>
</section>

<style>
  /* Bounds the pane so .view-pad's `flex:1;min-height:0;overflow-y:auto` is an
     actual scroll container instead of a no-op on a block-level parent (see
     tools-view in views/tools/View.svelte for the same fix) - otherwise the
     mux shell's own `.mux-bd` scrolls the whole view as one rigid block and
     the table's `position:sticky` header goes along for the ride. */
  .plugins-view {
    height: 100%;
    min-height: 0;
    display: flex;
    flex-direction: column;
  }

  /* .view-pad/.row/.dim/.grow - the same simple-registry-screen shell the
     Tools/Skills screens use. */
  .view-pad {
    flex: 1;
    min-height: 0;
    overflow-y: auto;
    padding: var(--sp-4) var(--sp-5) 26px;
    display: grid;
    gap: var(--sp-4);
    align-content: start;
  }
  .row {
    display: flex;
    align-items: center;
    gap: var(--sp-2);
  }
  .row h2 {
    margin: 0;
    font: 600 var(--fs-lg) var(--font-ui);
    color: var(--tx0);
  }
  .dim {
    color: var(--tx3);
  }
  .mono {
    font-family: var(--font-mono);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }

  /* .t-table - same block every hand-rolled domain table in this app
     (e.g. connstates/SecTable) repeats scoped to itself, since tokens.css
     ships no global component classes. */
  .plg-tbl-wrap {
    overflow-x: auto;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
  }
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
  .t-table td {
    padding: 5px 10px;
    border-bottom: 1px solid var(--bd0);
    height: 34px;
    vertical-align: middle;
  }
  .t-table tbody tr:last-child td {
    border-bottom: 0;
  }
  .t-table tbody tr:hover {
    background: color-mix(in oklab, var(--bg3) 45%, transparent);
  }
  .t-table tbody tr.is-off {
    opacity: 0.55;
  }
  .t-table .nm {
    font-weight: 600;
    color: var(--tx0);
  }
  .stcell {
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }
  .t-chip {
    display: inline-flex;
    align-items: center;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font-size: var(--fs-xs);
    color: var(--tx1);
    white-space: nowrap;
  }

  /* .t-callout: --err is a same-recipe sibling of the --warn variant. */
  .plg-errs {
    display: grid;
    gap: 8px;
  }
  .t-callout {
    display: flex;
    gap: 9px;
    align-items: flex-start;
    padding: 9px 12px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    background: var(--bg1);
    font-size: var(--fs-sm);
    color: var(--tx2);
    line-height: 1.5;
  }
  .t-callout b {
    color: var(--tx1);
    font-weight: 600;
  }
  .t-callout--err {
    border-color: color-mix(in oklab, var(--st-err) 42%, var(--bd1));
    background: color-mix(in oklab, var(--st-err) 8%, var(--bg1));
    color: var(--st-err);
  }
  .t-callout--err :global(.ic) {
    color: var(--st-err);
    flex: none;
    margin-top: 1px;
  }
</style>
