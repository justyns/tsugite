<script lang="ts">
  // Rail footer / keystrip (.rail-ft).
  // Pinned to the bottom of the nav rail: settings entry, today's usage readout,
  // and the live connection chip.
  import Button from '$lib/components/buttons/Button.svelte';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Conn from '$lib/components/connstates/Conn.svelte';
  import { conn } from '$lib/stores/conn.svelte';
  import { toConnDisplay } from './connDisplay';
  import { TESTID } from '$lib/testids';

  let {
    onOpenSettings,
    collapsed = false,
    cost = '$0.00',
    tokens = '0',
    model,
  }: {
    onOpenSettings: () => void;
    /** Icons-only rail: drop the usage readout + settings label, keep the glyphs. */
    collapsed?: boolean;
    /** Today's cost, e.g. "$1.84". Live value arrives with usage data. */
    cost?: string;
    /** Today's token count, e.g. "412k". Live value arrives with usage data. */
    tokens?: string;
    /** Active model id, qualified as "anthropic:claude-sonnet-4-6". Absent when
     *  no session is selected. */
    model?: string;
  } = $props();

  const shortModel = $derived(model ? model.split(':').slice(-1)[0] : undefined);
</script>

<div class="rail-ft" class:is-collapsed={collapsed} data-testid={TESTID.keystrip}>
  {#if collapsed}
    <Button
      variant="ghost"
      size="sm"
      iconOnly
      onclick={onOpenSettings}
      data-testid={TESTID.settingsTrigger}
      aria-label="Settings"
    >
      {#snippet icon()}<Icon name="gear" />{/snippet}
    </Button>
    <Conn state={toConnDisplay(conn.status)} onRetry={() => location.reload()} />
  {:else}
    <span class="settings-row">
      <Button
        variant="ghost"
        size="sm"
        onclick={onOpenSettings}
        data-testid={TESTID.settingsTrigger}
      >
        {#snippet icon()}<Icon name="gear" />{/snippet}
        Settings<span class="t-kbd">&#8984;,</span>
      </Button>
    </span>

    <div class="rail-usage">
      today <b>{cost}</b> &middot; <b>{tokens}</b> tok
      {#if shortModel}
        <br />
        <span class="model">{shortModel}</span>
      {/if}
    </div>

    <Conn state={toConnDisplay(conn.status)} onRetry={() => location.reload()} />
  {/if}
</div>

<style>
  /* .rail-ft */
  .rail-ft {
    margin-top: auto;
    display: grid;
    gap: 8px;
    padding: 8px 8px 2px;
    border-top: 1px solid var(--bd0);
  }
  /* Icons-only rail: stack the settings glyph + a bare conn dot, centered. */
  .rail-ft.is-collapsed {
    justify-items: center;
    gap: 6px;
  }
  .rail-ft.is-collapsed :global(.t-conn) {
    gap: 0;
  }
  .rail-ft.is-collapsed :global(.t-conn > :not(.t-dot)) {
    display: none;
  }
  /* The expanded settings row reads as a full-width nav row, not a centered
     button (overrides the shared Button's centering). */
  .settings-row {
    display: contents;
  }
  .settings-row :global(.t-btn) {
    justify-content: flex-start;
    gap: 7px;
    width: 100%;
  }
  .rail-usage {
    font: 500 var(--fs-2xs) / 1.6 var(--font-mono);
    color: var(--tx3);
    overflow-wrap: anywhere;
  }
  .rail-usage b {
    color: var(--tx2);
    font-weight: 600;
  }
  /* Narrow: the rail is a bottom bar; usage + conn move to the top bar. */
  @media (max-width: 640px) {
    .rail-ft {
      display: none;
    }
  }
  /* Base .t-kbd is global (tokens.css); the keystrip pushes its chip to the end. */
  .t-kbd {
    margin-left: auto;
  }
</style>
