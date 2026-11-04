<script lang="ts">
  // Popover for the link chips inside `root`, whose markup comes from {@html}. A
  // touch tap opens it instead of following the link. Mount it inside a positioned box.
  import { clampIntoView } from '$lib/actions/clampIntoView';
  import { linkPreviews } from '$lib/stores/linkPreviews.svelte';
  import { TESTID } from '$lib/testids';

  let { root }: { root: HTMLElement | undefined } = $props();

  /** Milliseconds the pointer may spend between the chip and the popover. */
  const CLOSE_DELAY = 150;

  let pop = $state<HTMLElement>();
  let chip = $state<HTMLAnchorElement | null>(null);
  let pos = $state({ top: 0, left: 0 });
  let closeTimer: ReturnType<typeof setTimeout> | undefined;

  const preview = $derived(chip && linkPreviews.lookup(chip.getAttribute('href') ?? ''));

  const chipOf = (target: EventTarget | null) =>
    target instanceof Element ? target.closest<HTMLAnchorElement>('a.t-linkchip') : null;

  function show(el: HTMLAnchorElement) {
    clearTimeout(closeTimer);
    chip = el;
  }

  function hideSoon() {
    clearTimeout(closeTimer);
    closeTimer = setTimeout(() => (chip = null), CLOSE_DELAY);
  }

  $effect(() => {
    if (!root) return;
    const el = root;
    const onOver = (e: Event) => {
      const target = chipOf(e.target);
      if (target) show(target);
    };
    const onOut = (e: Event) => {
      if (chipOf(e.target)) hideSoon();
    };
    // Recorded on pointerdown because a tap's focus and mouseover open the popover before its click.
    let wasOpen = false;
    const onDown = (e: PointerEvent) => {
      wasOpen = chipOf(e.target) === chip;
    };
    const onClick = (e: MouseEvent) => {
      const target = chipOf(e.target);
      const { pointerType } = e as PointerEvent;
      if (target && (pointerType === 'touch' || pointerType === 'pen') && !wasOpen) {
        e.preventDefault();
        show(target);
      }
    };
    el.addEventListener('mouseover', onOver);
    el.addEventListener('mouseout', onOut);
    el.addEventListener('focusin', onOver);
    el.addEventListener('focusout', onOut);
    el.addEventListener('pointerdown', onDown);
    el.addEventListener('click', onClick);
    return () => {
      el.removeEventListener('mouseover', onOver);
      el.removeEventListener('mouseout', onOut);
      el.removeEventListener('focusin', onOver);
      el.removeEventListener('focusout', onOut);
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('click', onClick);
    };
  });

  $effect(() => {
    if (!chip) return;
    const onDown = (e: PointerEvent) => {
      const target = e.target as Node;
      if (!pop?.contains(target) && !chip?.contains(target)) chip = null;
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') chip = null;
    };
    window.addEventListener('pointerdown', onDown);
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('pointerdown', onDown);
      window.removeEventListener('keydown', onKey);
    };
  });

  $effect(() => () => clearTimeout(closeTimer));

  function place(el: HTMLElement) {
    const parent = el.offsetParent ?? document.documentElement;
    const p = parent.getBoundingClientRect();
    const c = chip!.getBoundingClientRect();
    pos = {
      top: c.bottom - p.top - parent.clientTop + parent.scrollTop + 6,
      left: c.left - p.left - parent.clientLeft + parent.scrollLeft,
    };
    requestAnimationFrame(() => clampIntoView(el));
  }
</script>

{#if chip && preview}
  {#key chip}
    <div
      class="lp-pop"
      role="dialog"
      tabindex="-1"
      aria-label={preview.title}
      data-testid={TESTID.chatLinkPreviewPopover}
      style:top="{pos.top}px"
      style:left="{pos.left}px"
      bind:this={pop}
      onpointerenter={() => clearTimeout(closeTimer)}
      onpointerleave={(e) => e.pointerType === 'mouse' && hideSoon()}
      use:place
    >
      <div class="lp-body">
        <div class="lp-title">
          {#if preview.icon}
            <span class="ic-mask" aria-hidden="true" style:--icon="url('{preview.icon}')"></span>
          {/if}
          {preview.title}
        </div>
        {#if preview.badges.length}
          <div class="lp-badges">
            {#each preview.badges as badge, i (i)}
              <span class="t-chip lp-badge" data-tone={badge.tone}>{badge.text}</span>
            {/each}
          </div>
        {/if}
        {#if preview.fields.length}
          <dl class="lp-fields">
            {#each preview.fields as field, i (i)}
              <dt>{field.label}</dt>
              <dd>{field.value}</dd>
            {/each}
          </dl>
        {/if}
      </div>
      <a class="lp-link" href={preview.url} target="_blank" rel="noopener">Open</a>
    </div>
  {/key}
{/if}

<style>
  .lp-pop {
    position: absolute;
    z-index: 60;
    width: min(360px, 82vw);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    box-shadow: var(--sh-2);
    overflow: hidden;
    font-size: var(--fs-sm);
    line-height: 1.45;
    overflow-wrap: anywhere;
  }
  .lp-body {
    display: grid;
    gap: 6px;
    padding: 9px 11px;
  }
  .lp-title {
    display: flex;
    align-items: baseline;
    gap: 6px;
    color: var(--tx0);
    font-weight: 600;
  }
  .lp-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
  }
  .lp-badge {
    color: var(--tone);
    max-width: 100%;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .lp-fields {
    display: grid;
    grid-template-columns: auto minmax(0, 1fr);
    gap: 2px 10px;
    margin: 0;
    font: 400 var(--fs-xs) var(--font-mono);
  }
  .lp-fields dt {
    color: var(--tx3);
  }
  .lp-fields dd {
    margin: 0;
    color: var(--tx1);
  }
  .lp-link {
    display: block;
    padding: 7px 11px;
    border-top: 1px solid var(--bd0);
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx2);
  }
  .lp-link:hover {
    color: var(--tx0);
  }
</style>
