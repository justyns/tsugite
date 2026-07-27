<script lang="ts">
  import { untrack, type Snippet } from 'svelte';
  import Scrim from '$lib/components/overlays/Scrim.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import { trapFocus, focusables } from '$lib/actions/trapFocus';

  // Decision dialog. Focus is trapped inside, Esc cancels, initial focus lands
  // on the safe action (mark it `data-autofocus`), and the destructive verb is
  // spelled out by the caller's footer - never "OK". Modals for decisions;
  // Drawers for inspection. Both ride the same scrim + motion tokens.
  let {
    open = false,
    onclose,
    title,
    tone = 'default',
    children,
    footer,
  }: {
    open?: boolean;
    onclose?: () => void;
    title: string;
    tone?: 'default' | 'danger';
    children: Snippet;
    footer?: Snippet;
  } = $props();

  const titleId = $props.id();

  let modalEl = $state<HTMLElement | null>(null);
  let restoreTo: HTMLElement | null = null;
  // Snapshot the initial open state once; the $effect below tracks transitions.
  let wasOpen = untrack(() => open);

  function focusInitial() {
    if (!modalEl) return;
    const target = modalEl.querySelector<HTMLElement>('[data-autofocus]') ?? focusables(modalEl)[0];
    target?.focus();
  }

  // Move focus in only on a genuine closed→open transition (so a specimen that
  // mounts already-open - the gallery - never steals focus), and restore it to
  // the opener on open→close.
  $effect(() => {
    if (open === wasOpen) return;
    if (open) {
      restoreTo = document.activeElement as HTMLElement | null;
      focusInitial();
    } else {
      restoreTo?.focus();
      restoreTo = null;
    }
    wasOpen = open;
  });

  // Escape only; the Tab wrap is owned by the shared `use:trapFocus` action.
  function onKeydown(event: KeyboardEvent) {
    if (event.key === 'Escape') {
      event.stopPropagation();
      onclose?.();
    }
  }
</script>

<Scrim {open} {onclose}>
  <div
    bind:this={modalEl}
    class="t-modal"
    role="dialog"
    aria-modal="true"
    aria-labelledby={titleId}
    tabindex="-1"
    onkeydown={onKeydown}
    use:trapFocus
  >
    <h3 id={titleId}>
      {#if tone === 'danger'}
        <Icon name="alert" class="ic--danger" />
      {/if}
      {title}
    </h3>
    <div class="mb">{@render children()}</div>
    {#if footer}
      <div class="fx">{@render footer()}</div>
    {/if}
  </div>
</Scrim>

<style>
  .t-modal {
    width: min(430px, 100%);
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    padding: 16px;
    display: grid;
    gap: 12px;
    animation: toastin var(--t-3) var(--ease);
  }
  .t-modal h3 {
    margin: 0;
    font: 600 var(--fs-lg) / 1.3 var(--font-ui);
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .mb {
    font-size: var(--fs-sm);
    color: var(--tx2);
    line-height: 1.55;
    text-wrap: pretty;
  }
  /* Body text is caller-provided (snippet), so reach its <code> globally but
     scoped under this modal's .mb. */
  .mb :global(code) {
    font-family: var(--font-mono);
    color: var(--tx1);
    background: var(--bg1);
    padding: 0 4px;
    border-radius: 3px;
  }
  .fx {
    display: flex;
    gap: 8px;
    justify-content: flex-end;
  }

  /* The danger glyph is a shared <Icon>; its base sizing/stroke comes from the
     global .ic in tokens.css. Only the danger tint stays here, made :global so it
     reaches the child <Icon>'s svg (a scoped rule wouldn't carry its hash). */
  .t-modal :global(.ic--danger) {
    color: var(--st-err);
  }

  @keyframes toastin {
    from {
      translate: 0 8px;
      opacity: 0;
    }
  }

  @media (prefers-reduced-motion: reduce) {
    .t-modal {
      animation-duration: 0.01ms;
    }
  }
</style>
