<script lang="ts">
  import Icon from './Icon.svelte';
  import { ICONS, type IconName } from './icons';

  const names = Object.keys(ICONS) as IconName[];
</script>

<div class="grid">
  {#each names as name (name)}
    <figure>
      <Icon {name} />
      <figcaption>{name}</figcaption>
    </figure>
  {/each}
</div>

<p class="note">
  filled variant (<code>filled</code>): <Icon name="play" filled />
  <Icon name="dot" filled />
  <Icon name="send" filled />
</p>

<style>
  .grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(76px, 1fr));
    gap: var(--sp-3);
  }
  /* icons are 13px by design; scale up in the demo so they're legible */
  .grid :global(.ic),
  .note :global(.ic) {
    width: 20px;
    height: 20px;
  }
  figure {
    margin: 0;
    display: grid;
    gap: var(--sp-2);
    justify-items: center;
    padding: var(--sp-3) var(--sp-2);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    color: var(--tx1);
  }
  figcaption {
    font-family: var(--font-mono);
    font-size: var(--fs-2xs);
    color: var(--tx3);
  }
  .note {
    margin-top: var(--sp-4);
    display: flex;
    align-items: center;
    gap: var(--sp-2);
    color: var(--tx2);
  }
  .note code {
    font-family: var(--font-mono);
    font-size: var(--fs-xs);
    color: var(--tx2);
  }
</style>
