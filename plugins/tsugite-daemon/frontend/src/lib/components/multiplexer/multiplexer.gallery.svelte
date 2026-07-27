<script lang="ts">
  import Pane from './Pane.svelte';
  import TabStrip, { type PaneTab } from './TabStrip.svelte';

  const noop = () => {};

  // Hero: matches the card exactly - active session (green) + a pinned inactive
  // one (peach, no close x, as the card draws it) + the new-tab button.
  const heroTabs: PaneTab[] = [
    { id: 'sse', label: 'sse backoff', state: 'busy' },
    { id: 'prune', label: 'backup prune', state: 'blocked', closable: false },
  ];

  // Every dot state, side by side.
  const stateTabs: PaneTab[] = [
    { id: 'busy', label: 'busy', state: 'busy' },
    { id: 'streaming', label: 'streaming', state: 'streaming' },
    { id: 'blocked', label: 'blocked', state: 'blocked' },
    { id: 'error', label: 'error', state: 'error' },
    { id: 'idle', label: 'idle', state: 'idle' },
    { id: 'done', label: 'done', state: 'done' },
  ];
</script>

<section data-testid="gallery-multiplexer" class="ts-gallery">
  <div class="variant variant--w600">
    <span class="vlabel">pane · tab strip chrome (card)</span>
    <Pane bordered label="refactor: sse reconnect backoff">
      {#snippet tabs()}
        <TabStrip tabs={heroTabs} activeId="sse" onSelect={noop} onClose={noop} onNew={noop} />
      {/snippet}
    </Pane>
  </div>

  <div class="variant variant--w520">
    <span class="vlabel">tab strip · dot states + new</span>
    <div class="framed">
      <TabStrip tabs={stateTabs} activeId="busy" onSelect={noop} onClose={noop} onNew={noop} />
    </div>
  </div>

  <div class="variant variant--w600">
    <span class="vlabel">pane bodies · message + terminal</span>
    <div class="stack">
      <Pane bordered label="sse backoff — message body">
        <div class="mini-msg">
          <span class="who">you</span>can you add jitter to the reconnect backoff?
        </div>
        <div class="mini-msg">
          <span class="who">tsugite</span>On it — capping at 30s with full jitter, then a test.
        </div>
      </Pane>
      <Pane bordered terminal label="nightly term — terminal body">
        <div>$ borg prune --keep-daily 14 --keep-weekly 8</div>
        <div>Keeping archive: forge-2026-07-13 Mon, 2026-07-13 02:00:11</div>
        <div>pruning archive: forge-2026-06-14 (1/3)</div>
      </Pane>
    </div>
  </div>

  <div class="variant variant--w600">
    <span class="vlabel">split grid · frameless panes (divider from the grid gap)</span>
    <div class="split-grid">
      <Pane label="left pane">
        <div class="mini-msg"><span class="who">tsugite</span>working…</div>
      </Pane>
      <Pane label="right pane">
        <div class="mini-msg"><span class="who">ops-runner</span>blocked on a question.</div>
      </Pane>
    </div>
  </div>
</section>

<style>
  .ts-gallery {
    display: grid;
    gap: var(--sp-5);
  }
  .variant {
    display: grid;
    gap: var(--sp-2);
  }
  .vlabel {
    font: 600 var(--fs-2xs) / 1 var(--font-mono);
    letter-spacing: 0.07em;
    text-transform: uppercase;
    color: var(--tx3);
  }
  .stack {
    display: grid;
    gap: var(--sp-3);
  }
  /* Standalone frame so a bare strip reads as one pane in the gallery. */
  .framed {
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
  }
  .split-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1px;
    background: var(--bd1);
    border: 1px solid var(--bd0);
    border-radius: var(--r-md);
    overflow: hidden;
    min-height: 132px;
  }

  .mini-msg {
    padding: 9px 12px;
    border-top: 1px solid color-mix(in oklab, var(--bd0) 55%, transparent);
    font-size: var(--fs-sm);
    color: var(--tx1);
    line-height: 1.55;
  }
  .mini-msg .who {
    font: 600 var(--fs-2xs) var(--font-mono);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--tx3);
    margin-right: 8px;
  }

  .variant--w600 {
    max-width: 600px;
  }
  .variant--w520 {
    max-width: 520px;
  }
</style>
