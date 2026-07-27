<script lang="ts">
  // Terminals context rail: the PTY list (live metrics) + the run bar. Each row
  // opens (or focuses) that terminal as a surface in the focused pane via
  // `onOpenTerminal`; starting a command spawns a pty and opens it. The list is
  // fed by the global terminal_state SSE stream (App's sink) plus a light metrics
  // poll here (the list endpoint merges bytes/lines/last-line live from the pty).
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import Input from '$lib/components/inputs/Input.svelte';
  import PaneState from '$lib/components/connstates/PaneState.svelte';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { terminals, type TerminalState } from '$lib/stores/terminals.svelte';
  import TerminalRow from './TerminalRow.svelte';

  const METRICS_POLL_MS = 2500;

  let {
    focusedTerminalId,
    onOpenTerminal,
  }: {
    focusedTerminalId: string | null;
    onOpenTerminal: (id: string) => void;
  } = $props();

  let now = $state(Date.now());
  let cmd = $state('');
  let creating = $state(false);

  const list = $derived(terminals.list);
  const stateOf = (id: string, fallback: TerminalState): TerminalState =>
    terminals.stateOf(id) ?? fallback;

  $effect(() => {
    void terminals.load();
    const tick = setInterval(() => (now = Date.now()), 1000);
    const poll = setInterval(() => void terminals.load(), METRICS_POLL_MS);
    return () => {
      clearInterval(tick);
      clearInterval(poll);
    };
  });

  async function runCommand(command: string) {
    const value = command.trim().replace(/^\/run\s+/, '');
    if (!value || creating) return;
    creating = true;
    try {
      const term = await terminals.create({ cmd: value });
      cmd = '';
      onOpenTerminal(term.id);
      toasts.push('ok', 'Terminal started', { body: `${value.slice(0, 44)} · ${term.id}` });
    } catch (err) {
      toasts.push('err', 'Run failed', { body: err instanceof Error ? err.message : String(err) });
    } finally {
      creating = false;
    }
  }
</script>

<div class="term-rail">
  <header class="rail-hd">
    <Icon name="term" size={13} />
    <span class="rail-title">terminals</span>
    <span class="t-badge" aria-label="{list.length} terminals">{list.length}</span>
  </header>

  <div class="term-list" role="listbox" aria-label="Terminals">
    {#if terminals.loading && list.length === 0}
      <div class="rail-pane"><PaneState kind="loading" lines={3} /></div>
    {:else if terminals.error && list.length === 0}
      <div class="rail-pane">
        <PaneState kind="error" title="Couldn't load terminals">
          {#snippet icon()}<Icon name="alert" />{/snippet}
          {#snippet hint()}{terminals.error}{/snippet}
          {#snippet actions()}
            <Button size="sm" onclick={() => terminals.load()}>
              {#snippet icon()}<Icon name="retry" />{/snippet}
              Retry
            </Button>
          {/snippet}
        </PaneState>
      </div>
    {:else if list.length === 0}
      <div class="rail-pane">
        <PaneState kind="empty" title="No terminals yet">
          {#snippet icon()}<Icon name="term" />{/snippet}
          {#snippet hint()}Run a command below to start one.{/snippet}
        </PaneState>
      </div>
    {:else}
      {#each list as term (term.id)}
        <TerminalRow
          {term}
          {now}
          st={stateOf(term.id, term.state)}
          isActive={term.id === focusedTerminalId}
          onSelect={() => onOpenTerminal(term.id)}
        />
      {/each}
    {/if}
  </div>

  <form
    class="run-bar"
    onsubmit={(e) => {
      e.preventDefault();
      void runCommand(cmd);
    }}
  >
    <Input bind:value={cmd} mono placeholder="run a command…" ariaLabel="Run a command" />
    <Button variant="pri" iconOnly type="submit" loading={creating} aria-label="Run command">
      {#snippet icon()}<Icon name="play" />{/snippet}
    </Button>
  </form>
</div>

<style>
  .term-rail {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    height: 100%;
    background: var(--bg1);
  }
  .rail-hd {
    display: flex;
    align-items: center;
    gap: 7px;
    padding: 9px 12px;
    border-bottom: 1px solid var(--bd0);
    color: var(--tx2);
    font: 600 var(--fs-sm) var(--font-mono);
    letter-spacing: 0.04em;
    flex: none;
  }
  .rail-hd :global(.ic) {
    color: var(--tx3);
  }
  .rail-title {
    flex: 1;
  }
  .t-badge {
    display: inline-grid;
    place-items: center;
    min-width: 17px;
    height: 16px;
    padding: 0 5px;
    border-radius: var(--r-sm);
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx2);
  }
  .term-list {
    overflow-y: auto;
    flex: 1;
    padding-bottom: 6px;
    min-height: 0;
  }
  .rail-pane {
    padding: 10px;
  }
  .run-bar {
    display: flex;
    gap: 6px;
    padding: 8px 10px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    flex: none;
  }
</style>
