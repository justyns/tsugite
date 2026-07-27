<script lang="ts">
  // Definition-list key/value primitive: uppercase mono term, right-aligned
  // value that ellipsizes rather than wrapping. `mono` matches usages like a
  // linked branch name or a secrets-store path.
  export type KvItem = { term: string; value: string; mono?: boolean };

  let { items }: { items: KvItem[] } = $props();
</script>

<dl class="t-kv">
  {#each items as item (item.term)}
    <dt>{item.term}</dt>
    <dd class:mono={item.mono}>{item.value}</dd>
  {/each}
</dl>

<style>
  .t-kv {
    margin: 0;
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 5px 14px;
    font-size: var(--fs-sm);
    align-items: baseline;
  }
  .t-kv dt {
    color: var(--tx3);
    font: 500 var(--fs-2xs) / 1.7 var(--font-mono);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .t-kv dd {
    margin: 0;
    color: var(--tx1);
    text-align: right;
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* .mono is a general utility; scoped locally here since it
     isn't ported to a shared stylesheet yet. */
  .mono {
    font-family: var(--font-mono);
  }
</style>
