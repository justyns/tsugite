<script lang="ts" module>
  export type TableColumn = {
    key: string;
    label: string;
    sortable?: boolean;
    align?: 'right';
    hideLabel?: boolean;
  };

  export type SortState = { key: string; dir: 'ascending' | 'descending' };
</script>

<script lang="ts" generics="T">
  // Table primitive: sticky mono headers with aria-sort, cool-tint selection,
  // dimmed "off" rows. The consumer's `cells` snippet renders each row's <td>s,
  // so the td rules and .c2/.c3/.mono/.tr utilities are :global() under .t-table.
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';

  let {
    columns,
    rows,
    key,
    cells,
    selected,
    off,
    rowTestid,
    sort = null,
    onSort,
    ariaLabel,
    testid,
  }: {
    columns: TableColumn[];
    rows: readonly T[];
    key: (row: T) => string | number;
    cells: Snippet<[T]>;
    selected?: (row: T) => boolean;
    off?: (row: T) => boolean;
    rowTestid?: (row: T) => string;
    sort?: SortState | null;
    onSort?: (key: string) => void;
    ariaLabel?: string;
    testid?: string;
  } = $props();

  // Kept as a plain function (not $derived) since it's evaluated per-column
  // inside the #each below; TS can't narrow `sort` through the `?.`
  // equality check alone, so branch on it explicitly here instead.
  function ariaSortFor(col: TableColumn): SortState['dir'] | 'none' {
    return sort && sort.key === col.key ? sort.dir : 'none';
  }
</script>

<table class="t-table" aria-label={ariaLabel} data-testid={testid}>
  <thead>
    <tr>
      {#each columns as col (col.key)}
        {#if col.sortable}
          <th scope="col" class="sortable" class:tr={col.align} aria-sort={ariaSortFor(col)}>
            <button type="button" onclick={() => onSort?.(col.key)}>
              {col.label}
              <Icon name="chev-d" size={9} />
            </button>
          </th>
        {:else if col.hideLabel}
          <th scope="col"><span class="vh">{col.label}</span></th>
        {:else}
          <th scope="col" class:tr={col.align}>{col.label}</th>
        {/if}
      {/each}
    </tr>
  </thead>
  <tbody>
    {#each rows as row (key(row))}
      {@const isSelected = selected?.(row)}
      {@const isOff = off?.(row)}
      <tr
        class:is-selected={isSelected}
        class:is-off={isOff}
        aria-selected={isSelected ? 'true' : undefined}
        aria-disabled={isOff ? 'true' : undefined}
        data-testid={rowTestid?.(row)}
      >
        {@render cells(row)}
      </tr>
    {/each}
  </tbody>
</table>

<style>
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
  .t-table th.sortable {
    cursor: pointer;
  }
  .t-table th.sortable:hover {
    color: var(--tx1);
  }
  /* Icon renders its own <svg class="ic"> inside its own component scope, so
     this rule needs :global() to reach it - size itself comes from the
     `size` prop, this only covers this call site's layout. */
  .t-table th :global(.ic) {
    vertical-align: -1px;
    margin-left: 2px;
  }
  /* button reset: the sortable <th> needs a real, keyboard-focusable control,
     but must look exactly like the plain-text header. */
  .t-table th.sortable button {
    all: unset;
    display: inline-flex;
    align-items: center;
    cursor: pointer;
    font: inherit;
    color: inherit;
    text-transform: inherit;
    letter-spacing: inherit;
  }
  /* `all: unset` above out-specifies the app-wide bare `:focus-visible`
     rule, which would otherwise strip the keyboard focus ring - restate it
     here at matching specificity. */
  .t-table th.sortable button:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
    border-radius: var(--r-sm);
  }
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
  .t-table :global(td) {
    padding: 5px 10px;
    border-bottom: 1px solid var(--bd0);
    height: 34px;
    vertical-align: middle;
    white-space: nowrap;
  }
  .t-table tbody tr {
    cursor: default;
  }
  .t-table tbody tr:hover {
    background: color-mix(in oklab, var(--bg3) 45%, transparent);
  }
  .t-table tbody tr.is-selected {
    background: color-mix(in oklab, var(--acc) 11%, transparent);
    box-shadow: inset 2px 0 0 0 var(--acc);
  }
  .t-table tbody tr.is-off {
    opacity: 0.55;
  }
  .t-table :global(.tr) {
    text-align: right;
  }
  .t-table :global(.c2) {
    color: var(--tx2);
  }
  .t-table :global(.c3) {
    color: var(--tx3);
  }
  .t-table :global(.mono) {
    font-family: var(--font-mono);
  }
</style>
