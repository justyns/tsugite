<script lang="ts">
  // Top bar (.appbar).
  // Brand + optional subtitle, theme control, command-palette trigger, and (on
  // narrow viewports, where the rail footer is hidden) the connection chip.
  import type { Snippet } from 'svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Conn from '$lib/components/connstates/Conn.svelte';
  import { conn } from '$lib/stores/conn.svelte';
  import { toConnDisplay } from './connDisplay';
  import { TESTID } from '$lib/testids';

  let {
    onOpenPalette,
    onOpenSettings,
    subtitle,
    cost,
    tokens,
  }: {
    /** Opens the command palette; the overlay itself lands in a later stage. */
    onOpenPalette: () => void;
    /** Opens settings. Shown here only on mobile, where the rail footer that
     *  hosts the settings entry on desktop is hidden. */
    onOpenSettings?: () => void;
    /** Faint mono strapline next to the wordmark (workspace/host, once wired). */
    subtitle?: Snippet;
    /** Today's cost/tokens; mirrored here on narrow viewports where the rail
     *  footer (their desktop home) is hidden. */
    cost?: string;
    tokens?: string;
  } = $props();
</script>

<header class="appbar" data-testid={TESTID.topbar}>
  <div class="brandmark">
    <!-- tsugite wordmark (brand asset, not a UI-registry glyph) -->
    <svg viewBox="0 0 16 16" aria-hidden="true"
      ><path d="M2 4.5h7v3H6.5v4H2z" /><path d="M14 11.5H7v-3h2.5v-4H14z" /></svg
    >
    tsugite
    {#if subtitle}<span class="ver">{@render subtitle()}</span>{/if}
  </div>

  <span class="conn-mobile">
    <Conn state={toConnDisplay(conn.status)} />
    {#if cost && tokens}
      <span class="usage-mobile">{cost} · {tokens} tok</span>
    {/if}
  </span>

  <div class="grow"></div>

  <span class="settings-mobile">
    <Button
      variant="ghost"
      size="sm"
      iconOnly
      onclick={() => onOpenSettings?.()}
      data-testid={TESTID.settingsTrigger}
      aria-label="Settings"
      title="Settings"
    >
      {#snippet icon()}<Icon name="gear" />{/snippet}
    </Button>
  </span>

  <Button
    variant="ghost"
    size="sm"
    onclick={onOpenPalette}
    data-testid={TESTID.paletteTrigger}
    aria-label="Command palette"
    title="Command palette"
  >
    {#snippet icon()}<Icon name="search" />{/snippet}
    <span class="t-kbd">&#8984;K</span>
  </Button>
</header>

<style>
  /* .appbar */
  .appbar {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 7px 14px;
    border-bottom: 1px solid var(--bd0);
    background: var(--bg1);
    flex: none;
    flex-wrap: wrap;
  }
  .brandmark {
    display: flex;
    align-items: center;
    gap: 9px;
    font-weight: 600;
    font-size: var(--fs-lg);
    letter-spacing: 0.01em;
  }
  .brandmark svg {
    width: 17px;
    height: 17px;
    fill: var(--brand);
  }
  .brandmark .ver {
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx3);
    letter-spacing: 0.04em;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  /* Conn + today's usage live in the rail footer on desktop; they surface here
     when narrow (the rail footer is hidden below 640px). */
  .conn-mobile {
    display: none;
    align-items: center;
    gap: var(--sp-2);
  }
  .usage-mobile {
    font: 400 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    white-space: nowrap;
  }
  /* Settings lives in the rail footer on desktop; it surfaces here on mobile,
     where that footer is hidden. */
  .settings-mobile {
    display: none;
  }
  /* Base .t-kbd is global (tokens.css); only the trigger's spacing is local. */
  .t-kbd {
    margin-left: 2px;
  }
  @media (max-width: 640px) {
    .conn-mobile {
      display: inline-flex;
    }
    .settings-mobile {
      display: inline-flex;
    }
    /* The theme seg + strapline yield space on phones (theme stays reachable
       from Settings). */
    .appbar {
      flex-wrap: nowrap;
    }
    .brandmark .ver {
      display: none;
    }
    .appbar :global(.t-seg) {
      display: none;
    }
  }
</style>
