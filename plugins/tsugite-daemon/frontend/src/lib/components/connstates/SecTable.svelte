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
  import Table, { type TableColumn } from '$lib/components/datadisplay/Table.svelte';

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

  const columns: TableColumn[] = [
    { key: 'name', label: 'name' },
    { key: 'value', label: 'value' },
    { key: 'scope', label: 'scope' },
    { key: 'actions', label: 'actions', hideLabel: true },
  ];
</script>

<div class="sec-tbl-wrap">
  <Table {columns} {rows} key={(row) => row.name} {ariaLabel}>
    {#snippet cells(row)}
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
    {/snippet}
  </Table>
</div>

<style>
  .sec-tbl-wrap {
    overflow-x: auto;
    border: 1px solid var(--bd0);
    border-radius: var(--r-lg);
  }
  .nm {
    font: 600 var(--fs-md) var(--font-mono);
    color: var(--tx0);
  }
  .sub {
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
