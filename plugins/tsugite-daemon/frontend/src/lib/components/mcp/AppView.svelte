<script lang="ts">
  // MCP App view - an `ui://` HTML resource (ext-apps, io.modelcontextprotocol/ui).
  // The host renders a sandboxed iframe; init → ready handshake; display modes
  // inline / fullscreen / pip; capability footer. Presentational + callback props;
  // the sandboxed body is supplied by the caller as a snippet.
  import type { Snippet } from 'svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Spin from '$lib/components/feedback/Spin.svelte';
  import type { IconName } from '$lib/components/icon/icons';

  type AppMode = 'inline' | 'fullscreen' | 'pip';
  type AppLife = 'init' | 'ready';
  type Capability = { icon: IconName; label: string };

  let {
    name,
    source,
    iconChar,
    iconColor,
    mode = 'inline',
    border = true,
    life = 'ready',
    capabilities = [
      { icon: 'tool', label: 'call server tools' },
      { icon: 'send', label: 'send messages' },
      { icon: 'out', label: 'open links' },
    ],
    children,
    onMode,
    onReload,
    onBorderToggle,
  }: {
    name: string;
    source: string;
    iconChar: string;
    iconColor: string;
    mode?: AppMode;
    border?: boolean;
    life?: AppLife;
    capabilities?: Capability[];
    children?: Snippet;
    onMode?: (mode: AppMode) => void;
    onReload?: () => void;
    onBorderToggle?: () => void;
  } = $props();

  const modes: { mode: AppMode; icon: IconName; label?: string }[] = [
    { mode: 'inline', icon: 'app', label: 'inline' },
    { mode: 'fullscreen', icon: 'full' },
    { mode: 'pip', icon: 'pip' },
  ];
</script>

<div
  class="t-appview"
  class:is-full={mode === 'fullscreen'}
  data-life={life}
  data-border={border ? 'on' : 'off'}
>
  <div class="av-hd">
    <span class="av-ico" style={`--av-ico-bg:${iconColor}`}>{iconChar}</span>
    <span class="av-nm">{name}</span>
    <span class="av-src"><Icon name="plug" class="lk" />{source}</span>
    <div class="av-modes" role="group" aria-label="Display mode">
      {#each modes as m (m.mode)}
        <button
          type="button"
          class:is-active={mode === m.mode}
          aria-pressed={mode === m.mode}
          aria-label={m.mode}
          onclick={() => onMode?.(m.mode)}
          ><Icon name={m.icon} />{#if m.label}{m.label}{/if}</button
        >
      {/each}
    </div>
    <div class="av-tools">
      <button
        type="button"
        aria-label="Border preference"
        title="Border"
        onclick={() => onBorderToggle?.()}><Icon name="ring" /></button
      >
      <button type="button" aria-label="Reload" title="Reload" onclick={() => onReload?.()}
        ><Icon name="retry" /></button
      >
    </div>
  </div>

  <div class="av-init"><Spin />ui/initialize → handshake…</div>

  <div class="av-frame">
    <span class="av-tag"><Icon name="lock" />ui:// · sandboxed</span>
    <div class="av-body">
      {#if children}{@render children()}{/if}
    </div>
  </div>

  <div class="av-cap">
    <b>can:</b>
    {#each capabilities as cap (cap.label)}
      <span class="cap"><Icon name={cap.icon} />{cap.label}</span>
    {/each}
  </div>
</div>

<style>
  /* ===== MCP App view (owned here) ===== */
  .t-appview {
    border: 1px solid var(--bd1);
    border-radius: var(--r-lg);
    background: var(--bg1);
    overflow: hidden;
    max-width: 560px;
  }
  .t-appview .av-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 7px 10px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg2);
    flex-wrap: wrap;
  }
  .t-appview .av-ico {
    width: 22px;
    height: 22px;
    background: var(--av-ico-bg, var(--brand));
    border-radius: 6px;
    flex: none;
    display: grid;
    place-items: center;
    color: var(--on-brand);
    font: 700 11px system-ui;
  }
  .t-appview .av-nm {
    font: 600 var(--fs-sm) var(--font-ui);
    color: var(--tx0);
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .t-appview .av-src {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-appview .av-src :global(.lk) {
    width: 10px;
    height: 10px;
  }
  .t-appview .av-modes {
    display: inline-flex;
    background: var(--bg1);
    border: 1px solid var(--bd1);
    border-radius: var(--r-md);
    padding: 2px;
    gap: 2px;
    margin-left: auto;
  }
  .t-appview .av-modes button {
    border: 0;
    background: none;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
    padding: 3px 6px;
    border-radius: var(--r-sm);
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 4px;
  }
  .t-appview .av-modes button :global(.ic) {
    width: 11px;
    height: 11px;
  }
  .t-appview .av-modes button.is-active {
    background: var(--bg3);
    color: var(--tx0);
  }
  .t-appview .av-tools {
    display: inline-flex;
    gap: 2px;
  }
  .t-appview .av-tools button {
    width: 24px;
    height: 24px;
    display: grid;
    place-items: center;
    border: 0;
    background: none;
    color: var(--tx3);
    border-radius: var(--r-sm);
    cursor: pointer;
  }
  .t-appview .av-tools button:hover {
    background: var(--bg3);
    color: var(--tx1);
  }
  .t-appview .av-frame {
    position: relative;
    margin: 10px;
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    background: var(--bg0);
    min-height: 60px;
    overflow: hidden;
  }
  .t-appview[data-border='off'] .av-frame {
    border-color: transparent;
    background: transparent;
    margin: 8px 10px;
  }
  .t-appview .av-frame > .av-tag {
    position: absolute;
    top: 0;
    right: 0;
    font: 600 9px / 1 var(--font-mono);
    color: var(--tx3);
    background: var(--bg1);
    border-bottom-left-radius: var(--r-sm);
    padding: 3px 6px;
    display: inline-flex;
    gap: 4px;
    align-items: center;
    z-index: 2;
  }
  .t-appview .av-frame > .av-tag :global(.ic) {
    width: 9px;
    height: 9px;
    color: var(--st-ok);
  }
  .t-appview .av-body {
    padding: 12px;
  }
  .t-appview .av-init {
    display: none;
    padding: 22px 12px;
    place-items: center;
    gap: 8px;
    text-align: center;
    color: var(--tx3);
    font: 500 var(--fs-2xs) var(--font-mono);
  }
  .t-appview[data-life='init'] .av-init {
    display: grid;
  }
  .t-appview[data-life='init'] .av-body {
    display: none;
  }
  .t-appview .av-cap {
    display: flex;
    flex-wrap: wrap;
    gap: 6px 12px;
    padding: 7px 12px;
    border-top: 1px solid var(--bd0);
    background: var(--bg2);
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
  }
  .t-appview .av-cap b {
    color: var(--tx2);
    font-weight: 600;
  }
  .t-appview .av-cap .cap {
    display: inline-flex;
    gap: 5px;
    align-items: center;
  }
  .t-appview .av-cap .cap :global(.ic) {
    width: 10px;
    height: 10px;
    color: var(--tx3);
  }
  .t-appview.is-full {
    max-width: none;
  }
</style>
