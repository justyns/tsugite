<script lang="ts" module>
  import type { GitStatus, RefKind } from './types';
  import type { IconName } from '$lib/components/icon/icons';

  const ICON: Record<RefKind, IconName> = {
    file: 'file',
    chat: 'chat',
    agent: 'agent',
    terminal: 'term',
    session: 'chat',
    plugin: 'plug',
  };
  const GIT_LETTER: Record<GitStatus, string> = { m: 'M', a: 'A', u: '?', d: 'D' };
  const GIT_TITLE: Record<GitStatus, string> = {
    m: 'modified',
    a: 'added',
    u: 'untracked',
    d: 'deleted',
  };

  export interface RefNav {
    /** Next roving-highlight index (clamped to the list). */
    activeIndex: number;
    /** The active item should be committed. */
    select: boolean;
    /** The popover should close. */
    close: boolean;
    /** The key was consumed and should not reach the textarea. */
    handled: boolean;
  }

  /**
   * Pure keyboard-nav reducer for the reference popover. The composer owns the
   * textarea (and therefore focus), so it calls this on keydown and applies the
   * result - keeping the nav logic here, next to the popover, and unit-testable.
   */
  export function refNav(key: string, activeIndex: number, count: number): RefNav {
    const idle: RefNav = { activeIndex, select: false, close: false, handled: false };
    if (count === 0) return idle;
    const done = (i: number, extra: Partial<RefNav> = {}): RefNav => ({
      activeIndex: i,
      select: false,
      close: false,
      handled: true,
      ...extra,
    });
    switch (key) {
      case 'ArrowDown':
        return done(Math.min(activeIndex + 1, count - 1));
      case 'ArrowUp':
        return done(Math.max(activeIndex - 1, 0));
      case 'Home':
        return done(0);
      case 'End':
        return done(count - 1);
      case 'Enter':
      case 'Tab':
        return done(activeIndex, { select: true, close: true });
      case 'Escape':
        return done(activeIndex, { close: true });
      default:
        return idle;
    }
  }
</script>

<script lang="ts">
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { RefItem } from './types';

  let {
    items,
    activeIndex = $bindable(0),
    open = true,
    floating = false,
    idBase = 'ref-opt',
    listId,
    label = 'Reference suggestions',
    status,
    onSelect,
  }: {
    items: RefItem[];
    /** Roving highlight index; two-way so the composer can drive it from the textarea. */
    activeIndex?: number;
    open?: boolean;
    /** Absolutely position above the textarea (composer) vs. inline (card/standalone). */
    floating?: boolean;
    idBase?: string;
    /** Listbox element id, referenced by the composer textarea's aria-controls. */
    listId?: string;
    label?: string;
    /** Shown as a single non-interactive row when there are no items (a plugin
     *  source's "Searching…" / "No matches"); ignored once results arrive. */
    status?: string;
    onSelect: (item: RefItem, index: number) => void;
  } = $props();

  function choose(i: number) {
    const item = items[i];
    if (item) onSelect(item, i);
  }

  // The list is height-capped, so keep the roving-highlighted row in view as the
  // composer drives activeIndex from the textarea (arrow nav past the fold).
  let listEl = $state<HTMLDivElement>();
  $effect(() => {
    void activeIndex;
    if (!open || !listEl) return;
    listEl.querySelector<HTMLElement>('.is-hl')?.scrollIntoView({ block: 'nearest' });
  });
</script>

<div
  bind:this={listEl}
  class="slashpop"
  class:is-open={open}
  class:is-floating={floating}
  id={listId}
  role="listbox"
  aria-label={label}
>
  <!-- Highlight tracks real pointer motion (mousemove), not mouseenter: this
       popover floats above the textarea and can appear under a stationary
       cursor, which fires a synthesized mouseenter that would otherwise hijack
       the keyboard/auto highlight to whatever option lands under the pointer. -->
  {#if items.length === 0 && status}
    <div class="statusrow" role="presentation">{status}</div>
  {/if}
  {#each items as item, i (item.id)}
    <!-- Section header when the group changes; non-interactive, so the roving
         index (over items) stays aligned and arrow-nav skips it. -->
    {#if item.group && item.group !== items[i - 1]?.group}
      <div class="grouphead" role="presentation">{item.group}</div>
    {/if}
    <button
      type="button"
      role="option"
      id={`${idBase}-${i}`}
      class:is-hl={i === activeIndex}
      aria-selected={i === activeIndex}
      tabindex="-1"
      onclick={() => choose(i)}
      onmousemove={() => (activeIndex = i)}
    >
      <Icon name={ICON[item.kind]} />
      {item.label}
      {#if item.detail}<span class="d">{item.detail}</span>{/if}
      {#if item.git}
        <span class="git-m" data-g={item.git} title={`git: ${GIT_TITLE[item.git]}`}
          >{GIT_LETTER[item.git]}</span
        >
      {/if}
    </button>
  {/each}
</div>

<style>
  /* composer popover */
  .slashpop {
    width: min(320px, 100%);
    max-height: 40vh;
    overflow-y: auto;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    padding: 4px;
    display: none;
  }
  .slashpop.is-open {
    display: block;
  }
  .slashpop.is-floating {
    position: absolute;
    bottom: calc(100% + 6px);
    left: 0;
    z-index: 6;
  }
  .slashpop button {
    display: flex;
    gap: 10px;
    align-items: baseline;
    width: 100%;
    text-align: left;
    background: none;
    border: 0;
    padding: 6px 8px;
    border-radius: var(--r-sm);
    cursor: pointer;
    font: 500 var(--fs-sm) var(--font-mono);
    color: var(--tx0);
  }
  .slashpop button:hover,
  .slashpop button.is-hl {
    background: var(--bg3);
  }
  .slashpop button:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .slashpop .d {
    font: 400 var(--fs-xs) var(--font-ui);
    color: var(--tx3);
  }

  /* Non-interactive section header + status row: muted, uppercase, unclickable. */
  .grouphead,
  .statusrow {
    padding: 5px 8px 2px;
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: var(--tx3);
    user-select: none;
  }
  .statusrow {
    text-transform: none;
    letter-spacing: 0;
    font-weight: 400;
  }

  /* git file-state glyph - kept inline: no shared component owns it (only the CSS
     idea is shared with a future FileRow's `.wk-file .git-m`), so there is nothing
     to dedupe onto yet. */
  .git-m {
    margin-left: auto;
    align-self: center;
    flex: none;
    font: 700 var(--fs-2xs) / 1 var(--font-mono);
    padding: 1px 4px;
    border-radius: 3px;
  }
  .git-m[data-g='m'] {
    color: var(--st-warn);
    background: color-mix(in oklab, var(--st-warn) 15%, transparent);
  }
  .git-m[data-g='a'] {
    color: var(--st-ok);
    background: color-mix(in oklab, var(--st-ok) 15%, transparent);
  }
  .git-m[data-g='u'] {
    color: var(--tx3);
    background: var(--bg3);
  }
  .git-m[data-g='d'] {
    color: var(--st-err);
    background: color-mix(in oklab, var(--st-err) 15%, transparent);
  }
</style>
