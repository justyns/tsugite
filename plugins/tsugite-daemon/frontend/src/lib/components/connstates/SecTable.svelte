<script lang="ts" module>
  export type SecretRow = {
    name: string;
    /** e.g. "process env" */
    provenance: string;
    /** e.g. "env" */
    scope: string;
  };
</script>

<script lang="ts">
  // Write-only secrets table.
  // A value is never shown again after it's saved - only rotated or deleted.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';

  let {
    rows,
    ariaLabel = 'Secrets',
    onRotate,
    onDelete,
  }: {
    rows: SecretRow[];
    ariaLabel?: string;
    onRotate?: (row: SecretRow) => void;
    onDelete?: (row: SecretRow) => void;
  } = $props();
</script>

<div class="sec-tbl-wrap">
  <table class="t-table sec-tbl" aria-label={ariaLabel}>
    <thead>
      <tr>
        <th>name</th>
        <th>value</th>
        <th>scope</th>
        <th><span class="vh">actions</span></th>
      </tr>
    </thead>
    <tbody>
      {#each rows as row (row.name)}
        <tr>
          <td>
            <div class="nm">{row.name}</div>
            <div class="sub">{row.provenance}</div>
          </td>
          <td><span class="sec-mask" aria-label="value hidden">••••••••</span></td>
          <td><span class="t-chip mono">{row.scope}</span></td>
          <td>
            <div class="sec-acts">
              <Button size="sm" onclick={() => onRotate?.(row)}>
                {#snippet icon()}<Icon name="retry" />{/snippet}Rotate
              </Button>
              <Button
                size="sm"
                iconOnly
                variant="ghost"
                aria-label="Delete {row.name}"
                onclick={() => onDelete?.(row)}
              >
                {#snippet icon()}<Icon name="x" />{/snippet}
              </Button>
            </div>
          </td>
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .vh {
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
  .mono {
    font-family: var(--font-mono);
  }
  .sec-tbl-wrap {
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
    white-space: nowrap;
  }
  .t-table tbody tr:hover {
    background: color-mix(in oklab, var(--bg3) 45%, transparent);
  }
  .sec-tbl .nm {
    font: 600 var(--fs-md) var(--font-mono);
    color: var(--tx0);
  }
  .sec-tbl .sub {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .sec-mask {
    font: 600 var(--fs-lg)/1 var(--font-mono);
    color: var(--tx3);
    letter-spacing: 0.18em;
  }
  .sec-acts {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
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
    cursor: default;
  }
  .t-chip :global(.ic) {
    color: var(--tx3);
  }
</style>
