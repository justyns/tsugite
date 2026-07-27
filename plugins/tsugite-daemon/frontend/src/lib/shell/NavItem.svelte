<script lang="ts">
  // One primary-nav row (.t-nav).
  // A real anchor so the hash router drives it (Cmd-click, middle-click work);
  // the active row is marked by aria-current, never by the accent bar alone.
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import { TESTID } from '$lib/testids';

  let {
    id,
    label,
    icon,
    active = false,
    collapsed = false,
    badge,
    onactivate,
  }: {
    id: string;
    label: string;
    icon: IconName;
    active?: boolean;
    /** Icons-only rail: the label hides, so the glyph carries the name + a tooltip. */
    collapsed?: boolean;
    /** Live-count badges land here once session data is wired; omitted renders clean. */
    badge?: Snippet;
    /** Plain-click activation drives the mux; omit to leave the anchor a pure hash link. */
    onactivate?: (id: string) => void;
  } = $props();

  function onclick(event: MouseEvent) {
    // A plain click opens the view in the focused pane; modified / non-primary
    // clicks keep the anchor's native deep-link + new-tab behaviour.
    if (!onactivate) return;
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button !== 0)
      return;
    event.preventDefault();
    onactivate(id);
  }
</script>

<li>
  <a
    class="t-nav"
    class:is-active={active}
    class:is-collapsed={collapsed}
    href="#{id}"
    data-testid={TESTID.navTab(id)}
    aria-current={active ? 'page' : undefined}
    aria-label={collapsed ? label : undefined}
    title={collapsed ? label : undefined}
    {onclick}
  >
    <Icon name={icon} />
    <span class="lb">{label}</span>
    {#if badge}<span class="bdg">{@render badge()}</span>{/if}
  </a>
</li>

<style>
  li {
    list-style: none;
  }
  /* .t-nav */
  .t-nav {
    display: flex;
    align-items: center;
    gap: 8px;
    height: 29px;
    padding: 0 8px;
    border-radius: var(--r-md);
    color: var(--tx1);
    font: 500 var(--fs-md) / 1 var(--font-ui);
    cursor: pointer;
    text-decoration: none;
    position: relative;
  }
  .t-nav :global(.ic) {
    width: 14px;
    height: 14px;
    color: var(--tx3);
  }
  .t-nav:hover {
    background: var(--bg3);
    color: var(--tx0);
    text-decoration: none;
  }
  .t-nav.is-active {
    background: color-mix(in oklab, var(--acc) 13%, transparent);
    color: var(--tx0);
    box-shadow: inset 2px 0 0 0 var(--acc);
  }
  .t-nav.is-active :global(.ic) {
    color: var(--acc);
  }
  /* Icons-only: center the glyph, drop the label + badge (the anchor carries an
     aria-label + title so the row keeps its name and gains a tooltip). */
  .t-nav.is-collapsed {
    justify-content: center;
    padding: 0;
  }
  .t-nav.is-collapsed .lb,
  .t-nav.is-collapsed .bdg {
    display: none;
  }
  .lb {
    min-width: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .bdg {
    margin-left: auto;
    display: flex;
    gap: 4px;
    align-items: center;
  }

  /* Narrow: stack glyph over label, badge floats to the corner. */
  @media (max-width: 640px) {
    .t-nav {
      flex-direction: column;
      gap: 3px;
      height: auto;
      padding: 5px 9px;
      font-size: var(--fs-2xs);
    }
    .t-nav :global(.ic) {
      width: 16px;
      height: 16px;
    }
    .t-nav.is-active {
      box-shadow: none;
    }
    .bdg {
      position: absolute;
      top: 1px;
      right: 2px;
      margin: 0;
    }
  }
</style>
