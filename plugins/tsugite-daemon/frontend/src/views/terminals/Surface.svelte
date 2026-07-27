<script lang="ts">
  // Terminal surface: one xterm canvas bound to a pty, docked as a mux tab. The
  // pty list lives in the shared context rail (TerminalsRail); this surface shows
  // whichever terminal it's pointed at by `params.terminalId`. Restart spawns a
  // fresh pty id, so the bound id is navigable internal state (seeded from params)
  // rather than the param alone.
  import Icon from '$lib/components/icon/Icon.svelte';
  import { terminals, type TerminalState } from '$lib/stores/terminals.svelte';
  import { shellView } from '$lib/stores/shellView.svelte';
  import { goBackToWorkspaceList } from '$lib/shell/phoneNav';
  import PhoneBack from '$lib/shell/PhoneBack.svelte';
  import TerminalCanvas from './TerminalCanvas.svelte';

  let { params }: { params?: Record<string, string> } = $props();

  // Phone drilldown back: the terminal content screen returns to the pty list.
  const back = () => goBackToWorkspaceList('terminals');

  let terminalId = $state<string | null>(params?.terminalId ?? null);
  let now = $state(Date.now());

  // A rail click retargets this tab in place (spaces.openReusing rewrites the
  // tab's params); follow the pointed-at terminal. Internal rebinds (restart's
  // fresh pty id) keep working because they don't touch the param value.
  const paramTerminalId = $derived(params?.terminalId);
  $effect(() => {
    if (paramTerminalId) terminalId = paramTerminalId;
  });

  const list = $derived(terminals.list);
  const selected = $derived(list.find((t) => t.id === terminalId) ?? null);
  const stateOf = (id: string, fallback: TerminalState): TerminalState =>
    terminals.stateOf(id) ?? fallback;

  $effect(() => {
    if (list.length === 0) void terminals.load();
    const tick = setInterval(() => (now = Date.now()), 1000);
    return () => clearInterval(tick);
  });
</script>

<div class="term-surface">
  {#if selected}
    <TerminalCanvas
      term={selected}
      {now}
      st={stateOf(selected.id, selected.state)}
      onSelectTerminal={(id) => (terminalId = id)}
      onToggleRail={() => shellView.toggleRail('terminals')}
      onBack={back}
    />
  {:else}
    <div class="term-empty">
      <!-- A dead-link content screen (terminalId not in the list) still needs a way
           back to the list at phone width; the canvas header carries its own. -->
      <div class="term-empty-back"><PhoneBack label="Back to terminals" onBack={back} /></div>
      <Icon name="term" size={26} />
      <p class="empty-t">Terminal not available</p>
      <p class="empty-s">Pick a terminal from the list, or start one with the run bar.</p>
    </div>
  {/if}
</div>

<style>
  .term-surface {
    flex: 1;
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    background: var(--bg0);
    container: term-shell / inline-size;
  }
  .term-empty {
    position: relative;
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 4px;
    padding: 24px;
    text-align: center;
    background: #14161f;
    color: var(--tx3);
  }
  .term-empty-back {
    position: absolute;
    top: 8px;
    left: 8px;
  }
  .term-empty :global(.ic) {
    color: var(--tx3);
    margin-bottom: 6px;
  }
  .empty-t {
    margin: 0;
    color: var(--tx1);
    font: 600 var(--fs-lg) var(--font-ui);
  }
  .empty-s {
    margin: 0;
    max-width: 42ch;
    font-size: var(--fs-sm);
    line-height: 1.5;
  }
</style>
