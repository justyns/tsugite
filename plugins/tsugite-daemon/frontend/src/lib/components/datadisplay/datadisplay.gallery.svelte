<script lang="ts">
  // Static demo of every datadisplay state/variant, side by side. No
  // interaction - each variant is its own pre-configured instance.
  import Table from './Table.svelte';
  import type { TableColumn, SortState } from './Table.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Kv from './Kv.svelte';
  import type { KvItem } from './Kv.svelte';
  import Meter from './Meter.svelte';
  import Spark from './Spark.svelte';
  import type { SparkPoint } from './Spark.svelte';

  const tableColumns: TableColumn[] = [
    { key: 'name', label: 'name' },
    { key: 'state', label: 'state' },
    { key: 'updated', label: 'updated', sortable: true },
  ];
  const descSort: SortState = { key: 'updated', dir: 'descending' };
  const ascSort: SortState = { key: 'updated', dir: 'ascending' };

  const stateRows = [
    { id: 1, name: 'usage-rollup', st: 'done', icon: 'check', pill: 'done', updated: '2m' },
    {
      id: 2,
      name: 'nightly-backup (selected)',
      st: 'errored',
      icon: 'x',
      pill: 'errored',
      updated: '6h',
      selected: true,
    },
    {
      id: 3,
      name: 'model-cache-warm (disabled)',
      st: 'idle',
      icon: 'ring',
      pill: 'off',
      updated: '3d',
      off: true,
    },
  ] as const;
  const plainRows = [
    { id: 1, name: 'inbox-triage', state: 'running', updated: 'now' },
    { id: 2, name: 'usage-rollup', state: 'done', updated: '4s' },
  ];

  const kvPlain: KvItem[] = [
    { term: 'assignee', value: 'you' },
    { term: 'priority', value: 'High' },
    { term: 'sprint', value: 'Sprint 24 · ends in 3d' },
  ];
  const kvMono: KvItem[] = [
    { term: 'store', value: '~/.tsugite/secrets.age', mono: true },
    { term: 'cipher', value: 'age · x25519 + scrypt', mono: true },
    { term: 'entries', value: '9 keys' },
  ];

  const sparkAllOk: SparkPoint[] = [7, 9, 6, 8, 7, 10, 6, 8, 7, 9].map((height) => ({ height }));
  const sparkFailed: SparkPoint[] = [8, 7, 9, 7, 10, 7, 8, 9, 10, 7].map((height, i) => ({
    height,
    status: i === 4 || i === 7 || i === 8 ? 'fail' : undefined,
  }));
  const sparkSkipped: SparkPoint[] = [7, 8, 5, 7, 9, 7, 8, 7, 9, 8].map((height, i) => ({
    height,
    status: i === 2 ? 'skip' : undefined,
  }));
  const sparkEmpty: SparkPoint[] = Array.from({ length: 10 }, () => ({
    height: 5,
    status: 'skip',
  }));
</script>

<section data-testid="gallery-datadisplay">
  <h3>Table</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">descending sort · normal / selected / off rows</p>
      <div class="frame">
        <Table
          ariaLabel="Table demo: descending sort with row states"
          columns={tableColumns}
          sort={descSort}
          rows={stateRows}
          key={(row) => row.id}
          selected={(row) => 'selected' in row}
          off={(row) => 'off' in row}
        >
          {#snippet cells(row)}
            <td>{row.name}</td>
            <td>
              <span class="t-pill" data-st={row.st}
                ><Icon name={row.icon} size={11} />{row.pill}</span
              >
            </td>
            <td class="c3 mono">{row.updated}</td>
          {/snippet}
        </Table>
      </div>
    </div>
    <div class="demo">
      <p class="lbl">ascending sort · no sortable-column icon on other headers</p>
      <div class="frame">
        <Table
          ariaLabel="Table demo: ascending sort"
          columns={tableColumns}
          sort={ascSort}
          rows={plainRows}
          key={(row) => row.id}
        >
          {#snippet cells(row)}
            <td>{row.name}</td>
            <td>{row.state}</td>
            <td class="c3 mono">{row.updated}</td>
          {/snippet}
        </Table>
      </div>
    </div>
  </div>

  <h3>Kv</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">plain values</p>
      <div class="frame pad"><Kv items={kvPlain} /></div>
    </div>
    <div class="demo">
      <p class="lbl">mono values</p>
      <div class="frame pad"><Kv items={kvMono} /></div>
    </div>
  </div>

  <h3>Meter</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">default</p>
      <div class="frame pad">
        <Meter
          value={9200}
          max={200000}
          label="Context 9.2k of 200k tokens"
          displayText="9.2k/200k"
        />
      </div>
    </div>
    <div class="demo">
      <p class="lbl">is-warn (near limit)</p>
      <div class="frame pad">
        <Meter
          value={185000}
          max={200000}
          label="Context 185k of 200k tokens"
          displayText="185k/200k"
          warn
        />
      </div>
    </div>
  </div>

  <h3>Spark</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">all ok</p>
      <div class="frame pad"><Spark points={sparkAllOk} label="last 10 runs: all ok" /></div>
    </div>
    <div class="demo">
      <p class="lbl">3 failed</p>
      <div class="frame pad"><Spark points={sparkFailed} label="last 10 runs: 3 failed" /></div>
    </div>
    <div class="demo">
      <p class="lbl">1 skipped</p>
      <div class="frame pad"><Spark points={sparkSkipped} label="last 10 runs: 1 skipped" /></div>
    </div>
    <div class="demo">
      <p class="lbl">no recent runs</p>
      <div class="frame pad"><Spark points={sparkEmpty} label="no recent runs" /></div>
    </div>
  </div>
</section>

<style>
  section {
    display: grid;
    gap: var(--sp-3);
  }
  h3 {
    margin: var(--sp-3) 0 0;
    font: 600 var(--fs-xs) / 1 var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--tx3);
  }
  h3:first-child {
    margin-top: 0;
  }
  .demo-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--sp-4);
    align-items: flex-start;
  }
  .demo {
    display: grid;
    gap: 6px;
    min-width: 220px;
  }
  .lbl {
    margin: 0;
    font: 400 var(--fs-2xs) var(--font-ui);
    color: var(--tx3);
  }
  .frame {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
  }
  .frame.pad {
    padding: var(--sp-3);
    background: var(--bg1);
  }
  /* .t-pill copied inline for the demo only - kept inline (not folded into
     the shared Pill component): Pill only models the runtime PillState vocabulary
     (idle/busy/streaming/compacting/interrupted), whereas these demo pills use a
     done/errored/idle vocabulary Pill cannot express. */
  :global(.t-pill) {
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
  :global(.t-pill[data-st='queued']) {
    --c: var(--st-queue);
  }
  :global(.t-pill[data-st='done']) {
    --c: var(--st-mute);
  }
  :global(.t-pill[data-st='errored']) {
    --c: var(--st-err);
  }
  :global(.t-pill[data-st='idle']) {
    --c: var(--st-mute);
  }
</style>
