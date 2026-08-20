<script lang="ts">
  // Primary nav rail (.rail.app-rail).
  // View rows driven by the registry, settings + usage + conn pinned at the
  // bottom via KeyStrip. Collapses to an icons-only rail (labels hidden, glyphs
  // keep their accessible name + a tooltip); on narrow viewports it reflows to a
  // bottom bar instead.
  import type { ViewDef } from '../../views';
  import Icon from '$lib/components/icon/Icon.svelte';
  import NavItem from './NavItem.svelte';
  import KeyStrip from './KeyStrip.svelte';
  import type { NavBadge } from './navBadges';
  import { TESTID } from '$lib/testids';

  let {
    views,
    activeId,
    badges = {},
    collapsed = false,
    narrow = false,
    onActivate,
    onToggleCollapsed,
    onOpenSettings,
    keystripCost,
    keystripTokens,
  }: {
    views: ViewDef[];
    activeId: string;
    badges?: Record<string, NavBadge[]>;
    /** Icons-only mode; labels hide but each glyph keeps its aria-label + tooltip. */
    collapsed?: boolean;
    narrow?: boolean;
    /** Opens the clicked view; forwarded to each NavItem. */
    onActivate?: (id: string) => void;
    onToggleCollapsed?: () => void;
    onOpenSettings: () => void;
    /** Today's cost/tokens, pre-formatted; forwarded to KeyStrip. */
    keystripCost?: string;
    keystripTokens?: string;
  } = $props();
</script>

<nav
  class="rail app-rail"
  class:is-collapsed={collapsed}
  aria-label="Primary"
  data-testid={TESTID.navRail}
>
  {#if onToggleCollapsed}
    <button
      type="button"
      class="rail-collapse"
      aria-label={collapsed ? 'Expand navigation' : 'Collapse navigation'}
      aria-pressed={collapsed}
      title={collapsed ? 'Expand navigation' : 'Collapse navigation'}
      onclick={onToggleCollapsed}
    >
      <Icon name="chev-r" />
    </button>
  {/if}
  <ul class="t-navlist">
    {#each views as view (view.id)}
      <NavItem
        id={view.id}
        label={view.label}
        icon={view.icon}
        active={view.id === activeId}
        badges={badges[view.id]}
        {collapsed}
        {narrow}
        onactivate={onActivate}
      />
    {/each}
  </ul>
  <KeyStrip {collapsed} {onOpenSettings} cost={keystripCost} tokens={keystripTokens} />
</nav>

<style>
  /* .rail / .app-rail */
  .rail {
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 10px 8px;
    border-right: 1px solid var(--bd0);
    background: var(--bg0);
    min-width: 0;
    position: relative;
  }
  .app-rail {
    width: 198px;
    flex: none;
    transition: width var(--t-2) var(--ease);
  }
  .app-rail.is-collapsed {
    width: 52px;
  }
  @media (prefers-reduced-motion: reduce) {
    .app-rail {
      transition: none;
    }
  }
  .rail-collapse {
    align-self: flex-end;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 26px;
    height: 26px;
    margin-bottom: 2px;
    border: 1px solid transparent;
    border-radius: var(--r-md);
    background: none;
    color: var(--tx3);
    cursor: pointer;
    flex: none;
  }
  .rail-collapse:hover {
    background: var(--bg3);
    color: var(--tx0);
  }
  .rail-collapse:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .is-collapsed .rail-collapse {
    align-self: center;
  }
  .rail-collapse :global(.ic) {
    width: 13px;
    height: 13px;
    rotate: 180deg;
  }
  /* Collapsed: the glyph points right (expand); expanded it points left (collapse). */
  .is-collapsed .rail-collapse :global(.ic) {
    rotate: 0deg;
  }
  .t-navlist {
    display: flex;
    flex-direction: column;
    gap: 1px;
    padding: 0;
    margin: 0;
    list-style: none;
  }

  /* Narrow: the rail drops to a fixed bottom bar of the first five views (the
     rest stay reachable through the command palette). Collapse is meaningless
     there, so the toggle hides. */
  @media (max-width: 640px) {
    .app-rail,
    .app-rail.is-collapsed {
      width: auto;
      order: 2;
      flex-direction: row;
      align-items: center;
      border-right: 0;
      border-top: 1px solid var(--bd0);
      padding: 5px 8px max(5px, env(safe-area-inset-bottom));
    }
    .rail-collapse {
      display: none;
    }
    .t-navlist {
      flex-direction: row;
      flex: 1;
      gap: 2px;
      justify-content: space-around;
    }
    .t-navlist > :global(li:nth-child(n + 6)) {
      display: none;
    }
  }
</style>
