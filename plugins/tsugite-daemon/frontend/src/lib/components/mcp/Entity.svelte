<script lang="ts">
  // Entity auto-link chiplet (plugin message-decorator). A matched pattern
  // (e.g. JIRA-1234) becomes a live chiplet: status is dot + colour + text,
  // never colour alone. Click opens an anchored detail panel (the inline-panel
  // popover). Presentational + callback prop.
  import Icon from '$lib/components/icon/Icon.svelte';
  import Kv from '$lib/components/datadisplay/Kv.svelte';

  type EntityStatus = 'working' | 'blocked' | 'done' | 'err';

  let {
    entityKey,
    statusLabel,
    status = 'working',
    title,
    assignee,
    priority,
    sprint,
    via,
    open = false,
    onOpen,
  }: {
    entityKey: string;
    statusLabel: string;
    status?: EntityStatus;
    title?: string;
    assignee?: string;
    priority?: string;
    sprint?: string;
    via?: string;
    open?: boolean;
    onOpen?: () => void;
  } = $props();

  let isOpen = $state(open);
  // Only pull focus into the popover when the user opened it (keyboard path);
  // a statically-opened instance must not steal focus on mount.
  let userOpened = $state(false);
  let wrap: HTMLElement | undefined = $state();
  let anchor: HTMLButtonElement | undefined = $state();
  let pop: HTMLElement | undefined = $state();

  const detailRows = $derived(
    [
      { term: 'assignee', value: assignee },
      { term: 'priority', value: priority },
      { term: 'sprint', value: sprint },
    ].filter((r): r is { term: string; value: string } => Boolean(r.value)),
  );

  // The popover status uses the full t-pill badge, whose data-st vocabulary
  // differs from the chiplet's short entity status.
  const pillStatus = $derived(
    { working: 'running', blocked: 'awaiting', done: 'done', err: 'errored' }[status],
  );

  function toggle() {
    isOpen = !isOpen;
    if (isOpen) {
      userOpened = true;
      onOpen?.();
    } else {
      userOpened = false;
    }
  }

  function close() {
    isOpen = false;
    userOpened = false;
  }

  $effect(() => {
    if (!isOpen) return;
    if (userOpened) pop?.focus();
    const onDoc = (e: PointerEvent) => {
      if (wrap && !wrap.contains(e.target as Node)) close();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        close();
        anchor?.focus();
      }
    };
    document.addEventListener('pointerdown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('pointerdown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  });
</script>

<span class="ent-wrap" bind:this={wrap}>
  <button
    type="button"
    class="t-entity"
    data-st={status}
    aria-haspopup="dialog"
    aria-expanded={isOpen}
    bind:this={anchor}
    onclick={toggle}
  >
    {entityKey}<span class="edot" aria-hidden="true"></span><span class="est">{statusLabel}</span>
  </button>

  {#if isOpen}
    <div
      class="ent-pop is-open"
      role="dialog"
      aria-label={`${entityKey} details`}
      tabindex="-1"
      bind:this={pop}
    >
      <div class="eph">
        <span class="key">{entityKey}</span>
        <span class="t-pill" data-st={pillStatus}><span class="ptxt">{statusLabel}</span></span>
      </div>
      {#if title}<div class="etitle">{title}</div>{/if}
      {#if detailRows.length}
        <Kv items={detailRows} />
      {/if}
      {#if via}
        <div class="evia">
          <Icon name="plug" />auto-linked by {via} plugin
        </div>
      {/if}
    </div>
  {/if}
</span>

<style>
  .ent-wrap {
    position: relative;
    display: inline-flex;
    vertical-align: baseline;
  }

  /* ===== entity chiplet (owned here) ===== */
  .t-entity {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    vertical-align: baseline;
    padding: 0 6px;
    height: 19px;
    border: 1px solid var(--bd1);
    border-radius: var(--r-full);
    background: var(--bg2);
    color: var(--tx1);
    font: 600 var(--fs-2xs) var(--font-mono);
    cursor: pointer;
    text-decoration: none;
    position: relative;
    white-space: nowrap;
  }
  .t-entity:hover {
    border-color: var(--tx3);
    background: var(--bg3);
  }
  .t-entity .edot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    flex: none;
    background: var(--st-mute);
  }
  .t-entity .est {
    font-weight: 500;
    color: var(--tx3);
  }
  .t-entity[data-st='working'] .edot {
    background: var(--st-ok);
  }
  .t-entity[data-st='working'] .est {
    color: var(--st-ok);
  }
  .t-entity[data-st='blocked'] .edot {
    background: var(--st-warn);
  }
  .t-entity[data-st='blocked'] .est {
    color: var(--st-warn);
  }
  .t-entity[data-st='done'] .edot {
    background: var(--st-mute);
  }
  .t-entity[data-st='err'] .edot {
    background: var(--st-err);
  }
  .t-entity[data-st='err'] .est {
    color: var(--st-err);
  }

  /* detail popover, anchored absolutely here so the component is self-contained
     (a shared overlay layer would be the place to dedupe this). */
  .ent-pop {
    position: absolute;
    top: calc(100% + 6px);
    left: 0;
    z-index: 260;
    display: none;
    width: min(300px, 92vw);
    background: var(--bg2);
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-3);
    padding: 11px;
    flex-direction: column;
    gap: 8px;
    text-align: left;
  }
  .ent-pop.is-open {
    display: flex;
  }
  .ent-pop:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 2px;
  }
  .ent-pop .eph {
    display: flex;
    align-items: center;
    gap: 7px;
  }
  .ent-pop .eph .key {
    font: 700 var(--fs-sm) var(--font-mono);
    color: var(--tx0);
  }
  /* .t-pill stays inline: its data-st vocabulary (running/awaiting/done/errored)
     is the entity-status set, not the shared Pill's PillState
     (idle/busy/streaming/compacting/interrupted), so Pill can't model it. */
  .ent-pop .t-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 8px 0 7px;
    border-radius: var(--r-full);
    font: 500 var(--fs-xs) / 1 var(--font-mono);
    letter-spacing: 0.02em;
    white-space: nowrap;
    color: var(--c);
    background: color-mix(in oklab, var(--c) 13%, transparent);
    border: 1px solid color-mix(in oklab, var(--c) 32%, transparent);
  }
  .ent-pop .t-pill[data-st='running'] {
    --c: var(--st-ok);
  }
  .ent-pop .t-pill[data-st='awaiting'] {
    --c: var(--st-warn);
  }
  .ent-pop .t-pill[data-st='done'] {
    --c: var(--st-mute);
  }
  .ent-pop .t-pill[data-st='errored'] {
    --c: var(--st-err);
  }
  .ent-pop .etitle {
    font: 500 var(--fs-sm) / 1.4 var(--font-ui);
    color: var(--tx1);
  }
  .ent-pop .evia {
    display: flex;
    align-items: center;
    gap: 5px;
    font: 500 9px var(--font-mono);
    color: var(--tx3);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border-top: 1px solid var(--bd0);
    padding-top: 7px;
  }
  .ent-pop .evia :global(.ic) {
    width: 10px;
    height: 10px;
  }
</style>
