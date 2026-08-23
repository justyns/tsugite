<script lang="ts">
  // One row of a chronological list: glyph, title + sub-line, a state pill and a
  // relative stamp.
  import Icon from '$lib/components/icon/Icon.svelte';
  import type { IconName } from '$lib/components/icon/icons';
  import type { ActivityStatus } from '$lib/stores/activity.svelte';

  let {
    icon,
    title,
    detail = '',
    label,
    tone = null,
    when,
    testid,
    onopen,
  }: {
    icon: IconName;
    title: string;
    detail?: string;
    /** The word the pill prints - state is never carried by the tint alone. */
    label: string;
    tone?: ActivityStatus | null;
    when: string;
    testid: string;
    onopen: () => void;
  } = $props();
</script>

<button type="button" class="row" data-testid={testid} onclick={onopen}>
  <span class="glyph"><Icon name={icon} /></span>
  <span class="body">
    <span class="title">{title}</span>
    {#if detail}<span class="sum">{detail}</span>{/if}
  </span>
  <span class="pill" data-st={tone}>{label}</span>
  <span class="when">{when}</span>
</button>

<style>
  .row {
    width: 100%;
    display: grid;
    grid-template-columns: 18px minmax(0, 1fr) auto auto;
    align-items: center;
    gap: 10px;
    padding: 7px 10px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg2);
    color: var(--tx1);
    text-align: left;
    cursor: pointer;
  }
  .row:hover {
    background: var(--bg4);
    border-color: var(--bd1);
  }
  .row:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  .glyph {
    display: inline-flex;
    color: var(--tx3);
  }
  .body {
    display: grid;
    min-width: 0;
    gap: 1px;
  }
  .title {
    font-size: var(--fs-sm);
    color: var(--tx0);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .sum {
    font-size: var(--fs-xs);
    color: var(--tx3);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .pill {
    padding: 1px 7px;
    border-radius: var(--r-full);
    border: 1px solid var(--bd1);
    font: 500 var(--fs-2xs) / 1.6 var(--font-mono);
    color: var(--tx2);
    white-space: nowrap;
  }
  .pill[data-st='ok'] {
    color: var(--st-ok);
    border-color: color-mix(in oklab, var(--st-ok) 45%, transparent);
  }
  .pill[data-st='error'] {
    color: var(--st-err);
    border-color: color-mix(in oklab, var(--st-err) 45%, transparent);
  }
  .pill[data-st='cancelled'] {
    color: var(--st-mute);
    border-color: color-mix(in oklab, var(--st-mute) 45%, transparent);
  }
  .pill[data-st='skipped'] {
    color: var(--st-warn);
    border-color: color-mix(in oklab, var(--st-warn) 45%, transparent);
  }
  .when {
    font-size: var(--fs-2xs);
    font-family: var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }

  @media (max-width: 640px) {
    .row {
      grid-template-columns: 18px minmax(0, 1fr) auto;
      row-gap: 2px;
    }
    .when {
      grid-column: 2 / -1;
    }
  }
</style>
