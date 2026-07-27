<script lang="ts">
  // "What links here" panel: each backlink is a wk-back button showing the
  // citing file and the snippet of prose that mentions this note.
  type Backlink = { file: string; snippet: string; href?: string };

  let {
    links,
    heading = 'Backlinks',
    onSelect,
  }: {
    links: Backlink[];
    heading?: string;
    onSelect?: (link: Backlink) => void;
  } = $props();
</script>

<section class="backlinks" aria-label={heading}>
  <h4>{heading}</h4>
  {#if links.length === 0}
    <p class="empty">Nothing links here yet.</p>
  {:else}
    <ul>
      {#each links as link (link.file + link.snippet)}
        <li>
          <button type="button" class="wk-back" onclick={() => onSelect?.(link)}>
            <span class="f">{link.file}</span>
            <span class="s">{link.snippet}</span>
          </button>
        </li>
      {/each}
    </ul>
  {/if}
</section>

<style>
  .backlinks {
    display: grid;
    gap: 6px;
  }
  h4 {
    margin: 0;
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  ul {
    list-style: none;
    margin: 0;
    padding: 0;
    display: grid;
    gap: 5px;
  }
  .empty {
    margin: 0;
    font: 400 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
  }
  .wk-back {
    display: grid;
    gap: 2px;
    padding: 6px 7px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg2);
    cursor: pointer;
    min-width: 0;
    text-align: left;
    font: inherit;
    color: inherit;
    width: 100%;
  }
  .wk-back:hover {
    border-color: var(--acc);
  }
  .wk-back .f {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .wk-back .s {
    font-size: var(--fs-2xs);
    color: var(--tx3);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
</style>
