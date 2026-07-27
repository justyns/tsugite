<script lang="ts">
  // `[[wikilink]]` inline affordance for workspace markdown. Resolves to a
  // page; a missing target reads red AND carries a visually-hidden "(missing
  // page)" so the state never rides on color alone. Renders as a link when a
  // href is known, otherwise a button that fires onNavigate.
  let {
    page,
    missing = false,
    href,
    onNavigate,
  }: {
    page: string;
    missing?: boolean;
    href?: string;
    onNavigate?: (page: string) => void;
  } = $props();
</script>

{#if href}
  <a
    class="wikilink"
    class:is-missing={missing}
    {href}
    title={missing ? 'Missing page' : undefined}
    onclick={() => onNavigate?.(page)}
  >
    [[{page}]]{#if missing}<span class="vh"> (missing page)</span>{/if}
  </a>
{:else}
  <button
    type="button"
    class="wikilink"
    class:is-missing={missing}
    title={missing ? 'Missing page' : undefined}
    onclick={() => onNavigate?.(page)}
  >
    [[{page}]]{#if missing}<span class="vh"> (missing page)</span>{/if}
  </button>
{/if}

<style>
  .wikilink {
    color: var(--brand);
    border-bottom: 1px dashed color-mix(in oklab, var(--brand) 55%, transparent);
    cursor: pointer;
    /* button reset so it reads as inline prose text */
    display: inline;
    background: none;
    border-top: 0;
    border-left: 0;
    border-right: 0;
    padding: 0;
    font: inherit;
    line-height: inherit;
  }
  .wikilink:hover {
    color: color-mix(in oklab, var(--brand) 75%, var(--tx0));
    text-decoration: none;
    border-bottom-style: solid;
  }
  .wikilink.is-missing {
    color: var(--st-err);
    border-bottom-color: color-mix(in oklab, var(--st-err) 55%, transparent);
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
</style>
