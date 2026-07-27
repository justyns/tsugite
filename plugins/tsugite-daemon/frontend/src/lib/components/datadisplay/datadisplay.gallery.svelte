<script lang="ts">
  // Static demo of every datadisplay state/variant, side by side. No
  // interaction - each variant is its own pre-configured instance.
  import Table from './Table.svelte';
  import type { TableColumn, SortState } from './Table.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Kv from './Kv.svelte';
  import type { KvItem } from './Kv.svelte';
  import Stat from './Stat.svelte';
  import StatGrid from './StatGrid.svelte';
  import type { StatItem } from './Stat.svelte';
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

  const statGrid: StatItem[] = [
    { value: '12', label: 'open' },
    { value: '4', label: 'in prog' },
    { value: '1', label: 'blocked', tone: 'warn' },
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
        {#snippet pillDone()}
          <span class="t-pill" data-st="done"><Icon name="check" size={11} />done</span>
        {/snippet}
        {#snippet pillErrored()}
          <span class="t-pill" data-st="errored"><Icon name="x" size={11} />errored</span>
        {/snippet}
        {#snippet pillOff()}
          <span class="t-pill" data-st="idle"><Icon name="ring" size={11} />off</span>
        {/snippet}
        {#snippet nameSelected()}
          nightly-backup <span class="c3 mono">(selected)</span>
        {/snippet}
        {#snippet nameOff()}
          model-cache-warm <span class="c3 mono">(disabled)</span>
        {/snippet}
        <Table
          ariaLabel="Table demo: descending sort with row states"
          columns={tableColumns}
          sort={descSort}
          rows={[
            {
              id: 1,
              cells: [
                { content: 'usage-rollup' },
                { content: pillDone },
                { content: '2m', tone: 'c3', mono: true },
              ],
            },
            {
              id: 2,
              selected: true,
              cells: [
                { content: nameSelected },
                { content: pillErrored },
                { content: '6h', tone: 'c3', mono: true },
              ],
            },
            {
              id: 3,
              off: true,
              cells: [
                { content: nameOff },
                { content: pillOff },
                { content: '3d', tone: 'c3', mono: true },
              ],
            },
          ]}
        />
      </div>
    </div>
    <div class="demo">
      <p class="lbl">ascending sort · no sortable-column icon on other headers</p>
      <div class="frame">
        <Table
          ariaLabel="Table demo: ascending sort"
          columns={tableColumns}
          sort={ascSort}
          rows={[
            {
              id: 1,
              cells: [
                { content: 'inbox-triage' },
                { content: 'running' },
                { content: 'now', tone: 'c3', mono: true },
              ],
            },
            {
              id: 2,
              cells: [
                { content: 'usage-rollup' },
                { content: 'done' },
                { content: '4s', tone: 'c3', mono: true },
              ],
            },
          ]}
        />
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

  <h3>Stat / StatGrid</h3>
  <div class="demo-row">
    <div class="demo">
      <p class="lbl">StatGrid · default + warn tone</p>
      <div class="frame pad"><StatGrid stats={statGrid} /></div>
    </div>
    <div class="demo">
      <p class="lbl">Stat · with delta</p>
      <div class="frame pad"><Stat value="42" label="done today" delta="+6 vs yesterday" /></div>
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
  /* row-annotation utilities used by the name-cell snippets above - the
     table's own .c3/.mono only cover cells it renders itself, not these
     demo-only snippet spans. */
  .c3 {
    color: var(--tx3);
  }
  .mono {
    font-family: var(--font-mono);
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
