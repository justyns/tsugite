<script lang="ts">
  // Primary nav rail (.rail.app-rail).
  // View rows driven by the registry, settings + usage + conn pinned at the
  // bottom via KeyStrip. Collapses to an icons-only rail (labels hidden, glyphs
  // keep their accessible name + a tooltip); on narrow viewports it reflows to a
  // bottom bar instead. Hovering it while collapsed peeks the labelled rail open
  // over the workspace.
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
    peeking = false,
    onActivate,
    onToggleCollapsed,
    onHoverStart,
    onHoverEnd,
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
    peeking?: boolean;
    /** Opens the clicked view; forwarded to each NavItem. */
    onActivate?: (id: string) => void;
    onToggleCollapsed?: () => void;
    onHoverStart?: () => void;
    onHoverEnd?: () => void;
    onOpenSettings: () => void;
    /** Today's cost/tokens, pre-formatted; forwarded to KeyStrip. */
    keystripCost?: string;
    keystripTokens?: string;
  } = $props();

  const peeked = $derived(collapsed && peeking);
  const iconsOnly = $derived(collapsed && !peeked);

  let root: HTMLElement | undefined = $state();

  function hoverStart() {
    if (collapsed) onHoverStart?.();
  }

  function hoverEnd() {
    if (collapsed) onHoverEnd?.();
  }

  function onFocusOut(event: FocusEvent) {
    const next = event.relatedTarget;
    if (next instanceof Node && root?.contains(next)) return;
    hoverEnd();
  }
</script>

<nav
  class="rail app-rail"
  class:is-collapsed={collapsed}
  class:is-peeking={peeked}
  aria-label="Primary"
  data-testid={TESTID.navRail}
  data-peeking={peeked ? '' : undefined}
  bind:this={root}
  onmouseenter={hoverStart}
  onmouseleave={hoverEnd}
  onfocusin={hoverStart}
  onfocusout={onFocusOut}
>
  <div class="rail-body">
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
          collapsed={iconsOnly}
          {narrow}
          onactivate={onActivate}
        />
      {/each}
    </ul>
    <KeyStrip collapsed={iconsOnly} {onOpenSettings} cost={keystripCost} tokens={keystripTokens} />
  </div>
</nav>

<style>
  /* .rail / .app-rail */
  .rail {
    display: flex;
    flex-direction: column;
    border-right: 1px solid var(--bd0);
    background: var(--bg0);
    min-width: 0;
    position: relative;
  }
  .rail-body {
    flex: 1;
    min-height: 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    gap: 2px;
    padding: 10px 8px;
  }
  .app-rail {
    width: 198px;
    flex: none;
    transition: width var(--t-2) var(--ease);
  }
  .app-rail.is-collapsed {
    width: 52px;
  }
  /* Peeking, the nav keeps its 52px slot in the shell row while the body floats
     over the workspace at full width. */
  .app-rail.is-peeking .rail-body {
    position: absolute;
    top: 0;
    bottom: 0;
    left: 0;
    width: 198px;
    z-index: 60;
    background: var(--bg0);
    border-right: 1px solid var(--bd0);
    box-shadow: var(--sh-2);
    animation: nav-peek-in var(--t-2) var(--ease);
  }
  @keyframes nav-peek-in {
    from {
      translate: -102% 0;
    }
    to {
      translate: 0 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .app-rail {
      transition: none;
    }
    .app-rail.is-peeking .rail-body {
      animation: none;
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
  .is-peeking .rail-collapse {
    align-self: flex-end;
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
      border-right: 0;
      border-top: 1px solid var(--bd0);
    }
    .rail-body {
      flex-direction: row;
      align-items: center;
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
