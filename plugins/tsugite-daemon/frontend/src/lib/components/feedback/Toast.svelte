<script lang="ts">
  // Single toast.
  // "Toasts announce state changes; anything requiring action links straight to
  // the owning surface. Errors persist until dismissed." Usually hosted by
  // <Toasts> (the stack), but stands alone fine (see the gallery).
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import {
    isSticky,
    AUTO_DISMISS_MS,
    RESUME_DISMISS_MS,
    EXIT_DURATION_MS,
    type ToastVariant,
    type ToastIcon,
  } from './toast-store.svelte';

  const DEFAULT_ICON: Record<ToastVariant, ToastIcon> = {
    ok: 'check',
    warn: 'alert',
    err: 'x',
    info: 'dot',
  };

  let {
    variant,
    title,
    body,
    icon,
    actionLabel,
    onAction,
    sticky = false,
    onDismiss,
  }: {
    variant: ToastVariant;
    title: string;
    body?: string;
    /** Overrides the variant's default icon (e.g. a job-question toast uses `q`). */
    icon?: ToastIcon;
    actionLabel?: string;
    onAction?: () => void;
    /** `err` is always sticky regardless of this flag - see isSticky(). */
    sticky?: boolean;
    onDismiss: () => void;
  } = $props();

  const resolvedIcon = $derived(icon ?? DEFAULT_ICON[variant]);
  const effectivelySticky = $derived(isSticky(variant, sticky));

  let out = $state(false);
  let dismissTimer: ReturnType<typeof setTimeout> | undefined;
  let exitTimer: ReturnType<typeof setTimeout> | undefined;

  function scheduleExit(delay: number): void {
    clearTimeout(dismissTimer);
    dismissTimer = setTimeout(requestExit, delay);
  }

  function requestExit(): void {
    if (out) return;
    out = true;
    exitTimer = setTimeout(onDismiss, EXIT_DURATION_MS);
  }

  function pause(): void {
    clearTimeout(dismissTimer);
  }

  function resume(): void {
    if (effectivelySticky || out) return;
    scheduleExit(RESUME_DISMISS_MS);
  }

  $effect(() => {
    if (!effectivelySticky) scheduleExit(AUTO_DISMISS_MS);
    return () => {
      clearTimeout(dismissTimer);
      clearTimeout(exitTimer);
    };
  });
</script>

<div
  class="t-toast"
  class:is-out={out}
  data-v={variant}
  role="group"
  onmouseenter={pause}
  onmouseleave={resume}
  onfocusin={pause}
  onfocusout={resume}
>
  <Icon name={resolvedIcon} />
  <div class="grow">
    <div class="tt">{title}</div>
    {#if body}
      <div class="tb">{body}</div>
    {/if}
    {#if actionLabel && onAction}
      <div class="ta">
        <Button variant="pri" size="sm" onclick={onAction}>{actionLabel}</Button>
      </div>
    {/if}
  </div>
  <button type="button" class="t-iconbtn" aria-label="Dismiss" onclick={() => requestExit()}>
    <Icon name="x" />
  </button>
</div>

<style>
  .t-toast {
    display: flex;
    gap: 10px;
    align-items: flex-start;
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-left: 3px solid var(--c, var(--acc));
    border-radius: var(--r-md);
    box-shadow: var(--sh-3);
    padding: 10px 12px;
    animation: toastin var(--t-3) var(--ease);
    /* Ideally `.t-toast .ic{color:var(--c)}`; Icon is a child component so
       scoped CSS can't reach its internals (see Work.svelte for the same
       issue) - set color here instead and let the icon's `currentColor`
       inherit it. .tt/.tb/.t-iconbtn all set their own color, so this only
       ever reaches the variant icon. */
    color: var(--c, var(--acc));
  }
  @keyframes toastin {
    from {
      translate: 0 8px;
      opacity: 0;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-toast {
      animation: none;
    }
  }
  .t-toast.is-out {
    opacity: 0;
    translate: 0 6px;
    transition:
      opacity var(--t-3),
      translate var(--t-3);
  }
  .t-toast[data-v='ok'] {
    --c: var(--st-ok);
  }
  .t-toast[data-v='warn'] {
    --c: var(--st-warn);
  }
  .t-toast[data-v='err'] {
    --c: var(--st-err);
  }
  .t-toast[data-v='info'] {
    --c: var(--st-info);
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .tt {
    font: 600 var(--fs-sm)/1.35 var(--font-ui);
    color: var(--tx0);
  }
  .tb {
    font: 400 var(--fs-xs)/1.5 var(--font-ui);
    color: var(--tx2);
    margin-top: 1px;
    text-wrap: pretty;
  }
  .ta {
    margin-top: 6px;
  }

  /* dismiss button - not part of the .t-btn family (no border/bg), so it
     stays a local one-off rather than moving to the shared Button component. */
  .t-iconbtn {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: none;
    border: 0;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    cursor: pointer;
    padding: 2px 4px;
    border-radius: var(--r-sm);
  }
  .t-iconbtn:hover {
    color: var(--tx0);
    background: var(--bg3);
  }
</style>
