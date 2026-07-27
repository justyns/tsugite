<script lang="ts">
  // Terminal pane: header + live xterm canvas + follow pill + status bar. xterm
  // and its fit addon are dynamically imported (kept out of the main bundle) the
  // first time a canvas mounts. The canvas is the sole stdin sink - keystrokes
  // in the focused canvas POST straight to the pty (no separate input box).
  // Server `state` and the client `follow` boolean are orthogonal
  // axes: a running terminal can have follow:false.
  import type { Terminal as XTerm, ITheme } from '@xterm/xterm';
  import type { FitAddon as FitAddonType } from '@xterm/addon-fit';
  import Icon from '$lib/components/icon/Icon.svelte';
  import Button from '$lib/components/buttons/Button.svelte';
  import PhoneBack from '$lib/shell/PhoneBack.svelte';
  import StaleStamp from '$lib/components/connstates/StaleStamp.svelte';
  import { navigate } from '$lib/router.svelte';
  import { formatElapsed } from '$lib/components/feedback/format';
  import { toasts } from '$lib/components/feedback/toast-store.svelte';
  import { terminals, type Terminal, type TerminalState } from '$lib/stores/terminals.svelte';
  import { attachRecordToChat, copyReference } from '../chats/attachRecord';
  import TermPill from './TermPill.svelte';
  import { elapsedSeconds, formatBytes, isLiveTerminal, terminalPill } from './termState';

  let {
    term,
    // Named `st`, not `state` (avoids the `$state` rune / local-binding clash).
    st,
    now,
    onSelectTerminal,
    onToggleRail,
    onBack,
  }: {
    term: Terminal;
    /** Resolved live state (store overlay wins over the record's own field). */
    st: TerminalState;
    now: number;
    /** Restart spawns a new id; the view re-selects it (the restart chain). */
    onSelectTerminal: (id: string) => void;
    /** Reveals the rail drawer when the pane is too narrow to show it inline. */
    onToggleRail: () => void;
    /** Phone drilldown: leave the terminal and return to the pty list. */
    onBack: () => void;
  } = $props();

  // Fixed dark terminal background - deliberately theme-independent, matching
  // `.term-out` (#14161f) so the canvas reads as a real
  // terminal in every app theme.
  const CANVAS_BG = '#14161f';
  const XTERM_THEME: ITheme = {
    background: CANVAS_BG,
    foreground: '#b8bdcc',
    cursor: '#cdd3e0',
    selectionBackground: 'rgba(120,130,170,0.35)',
    black: '#14161f',
    brightBlack: '#6b7089',
  };

  // Stable across metric-poll refreshes: the parent hands a fresh record object
  // every poll, but a `$derived` primitive only notifies when the id string
  // actually changes, so the canvas rebuild effect below fires on row switch
  // only - never on a same-terminal metrics refresh.
  const canvasId = $derived(term.id);
  const live = $derived(isLiveTerminal(st));
  const following = $derived(terminals.isFollowing(term.id));
  const queued = $derived(terminals.queuedLines[term.id] ?? 0);
  const statusWord = $derived(terminalPill(st, term.exit_code).label);
  const elapsed = $derived(formatElapsed(elapsedSeconds(term.created_at, term.resolved_at, now)));

  let host = $state<HTMLDivElement>();
  let xterm: XTerm | null = null;
  let streamConnected = $state(true);
  let focused = $state(false);
  let killArmed = $state(false);
  let killTimer: ReturnType<typeof setTimeout> | null = null;
  let restarting = $state(false);

  // (Re)build the canvas + stream whenever the selected terminal id changes.
  // The effect's cleanup disposes the previous xterm and closes its stream, so
  // switching rows never leaks a PTY connection.
  $effect(() => {
    const el = host;
    const id = canvasId;
    if (!el) return;

    let disposed = false;
    let handle: { close(): void } | null = null;
    let ro: ResizeObserver | null = null;
    let fit: FitAddonType | null = null;

    void (async () => {
      const [{ Terminal: XTermCtor }, { FitAddon }] = await Promise.all([
        import('@xterm/xterm'),
        import('@xterm/addon-fit'),
      ]);
      await import('@xterm/xterm/css/xterm.css');
      if (disposed) return;

      const t = new XTermCtor({
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        fontSize: 12,
        lineHeight: 1.25,
        scrollback: 5000,
        cursorBlink: false,
        theme: XTERM_THEME,
        convertEol: false,
      });
      fit = new FitAddon();
      t.loadAddon(fit);
      t.open(el);
      safeFit(fit);
      xterm = t;

      t.onData((data) => {
        // Keystrokes (and xterm's own device-query replies) go straight to the
        // pty - but only while it's alive; a dead terminal is read-only.
        if (!isLiveTerminal(terminals.stateOf(id) ?? st)) return;
        void terminals.stdin(id, data).catch(() => {});
      });
      t.onScroll(() => {
        const buf = t.buffer.active;
        terminals.setFollow(id, buf.viewportY >= buf.baseY);
      });
      t.textarea?.addEventListener('focus', () => (focused = true));
      t.textarea?.addEventListener('blur', () => (focused = false));

      ro = new ResizeObserver(() => fit && safeFit(fit));
      ro.observe(el);

      handle = terminals.stream(id, {
        onOutput: (chunk) => t.write(chunk),
        onStatus: (connected) => (streamConnected = connected),
      });
    })();

    return () => {
      disposed = true;
      handle?.close();
      ro?.disconnect();
      xterm?.dispose();
      xterm = null;
      streamConnected = true;
      focused = false;
      disarmKill();
    };
  });

  function safeFit(fit: FitAddonType) {
    try {
      fit.fit();
    } catch {
      // container not laid out yet; the ResizeObserver will retry
    }
  }

  function jumpToTail() {
    terminals.setFollow(term.id, true);
    xterm?.scrollToBottom();
  }

  function disarmKill() {
    killArmed = false;
    if (killTimer) {
      clearTimeout(killTimer);
      killTimer = null;
    }
  }

  async function onKill() {
    if (!killArmed) {
      // First click only arms - no request sent. Auto-disarm after 3s.
      killArmed = true;
      killTimer = setTimeout(disarmKill, 3000);
      return;
    }
    disarmKill();
    try {
      await terminals.kill(term.id);
      toasts.push('warn', 'Terminal killed', { body: `${term.cmd.slice(0, 44)} · record kept` });
    } catch (err) {
      toasts.push('err', 'Kill failed', { body: err instanceof Error ? err.message : String(err) });
    }
  }

  async function onRestart() {
    restarting = true;
    try {
      const next = await terminals.restart(term.id);
      toasts.push('ok', 'PTY restarted', { body: `${next.id} · restarted from ${term.id}` });
      onSelectTerminal(next.id);
    } catch (err) {
      toasts.push('err', 'Restart failed', {
        body: err instanceof Error ? err.message : String(err),
      });
    } finally {
      restarting = false;
    }
  }

  async function onCopy() {
    if (!xterm || !navigator.clipboard) return;
    xterm.selectAll();
    const text = xterm.getSelection();
    xterm.clearSelection();
    try {
      await navigator.clipboard.writeText(text);
      toasts.push('info', 'Copied', { body: `${term.lines_out} lines · ANSI stripped` });
    } catch {
      // clipboard blocked (permissions / insecure context) - silent
    }
  }
</script>

<section class="term-pane" aria-label="Terminal output">
  <header class="term-hd">
    <PhoneBack label="Back to terminals" {onBack} />
    <span class="termsb">
      <Button
        size="sm"
        variant="ghost"
        iconOnly
        aria-label="Show terminal list"
        onclick={onToggleRail}
      >
        {#snippet icon()}<Icon name="term" />{/snippet}
      </Button>
    </span>
    <span class="cmdline" title={term.cmd}>{term.cmd}</span>
    <TermPill {st} exitCode={term.exit_code} />
    {#if term.restarted_from}
      <span class="t-chip" title="restarted from a previous terminal">
        <Icon name="retry" />from {term.restarted_from}
      </span>
    {/if}
    {#if term.parent_session_id}
      <button
        type="button"
        class="t-chip sess-link"
        title="open the session that started this terminal"
        onclick={() => navigate('chats', { sessionId: term.parent_session_id! })}
      >
        <Icon name="agent" />{term.parent_session_id}<Icon name="out" size={9} />
      </button>
    {/if}
    {#if st === 'stream_lost'}
      <StaleStamp />
    {/if}
    <div class="grow"></div>
    <Button
      size="sm"
      variant="ghost"
      iconOnly
      aria-label="Add terminal to chat"
      onclick={() => void attachRecordToChat('terminal', term.id)}
    >
      {#snippet icon()}<Icon name="chat" />{/snippet}
    </Button>
    <Button
      size="sm"
      variant="ghost"
      iconOnly
      aria-label="Copy terminal reference"
      onclick={() => void copyReference('terminal', term.id)}
    >
      {#snippet icon()}<Icon name="link" />{/snippet}
    </Button>
    <Button size="sm" variant="ghost" onclick={onCopy}>
      {#snippet icon()}<Icon name="copy" />{/snippet}
      copy
    </Button>
    {#if live}
      <button
        type="button"
        class="t-btn t-btn--sm t-btn--danger"
        class:is-armed={killArmed}
        onclick={onKill}
      >
        <Icon name="stop" />
        {killArmed ? 'confirm kill?' : 'Kill'}
      </button>
    {:else}
      <Button size="sm" loading={restarting} onclick={onRestart}>
        {#snippet icon()}<Icon name="retry" />{/snippet}
        Restart
      </Button>
    {/if}
  </header>

  <div
    bind:this={host}
    class="term-out"
    role="log"
    aria-live="off"
    aria-label="Terminal canvas — keystrokes go to the pty"
  ></div>

  {#if !following}
    <button type="button" class="term-jump is-show" onclick={jumpToTail}>
      <Icon name="down" size={11} />
      {queued > 0
        ? `follow paused — ${queued} new line${queued === 1 ? '' : 's'}`
        : 'jump to latest'}
    </button>
  {/if}

  {#if killArmed}
    <p class="armed-warn" role="status">SIGTERM armed · click Kill again to force SIGKILL</p>
  {/if}

  <div class="term-status">
    <span>state <b>{statusWord}</b></span>
    <span>elapsed <b>{elapsed}</b></span>
    <span>pid <b>{term.pid ?? '—'}</b></span>
    <span><b>{term.lines_out}</b> lines · <b>{formatBytes(term.bytes_out)}</b></span>
    {#if term.cwd}<span class="cwd">cwd <b>{term.cwd}</b></span>{/if}
    <div class="grow"></div>
    <span>
      {#if live}
        <!-- A dropped stream on a live pty is a real reconnect; a terminated
             one closing its stream is expected, so only flag it while live. -->
        {!streamConnected ? 'reconnecting…' : focused ? '⌨ keys → pty' : 'click canvas to type'}
        · {following ? 'following tail' : 'follow paused'}
      {:else}
        exited · read-only
      {/if}
    </span>
  </div>
</section>

<style>
  /* .term-pane / .term-hd / .term-out / .term-jump / .term-status */
  .term-pane {
    display: flex;
    flex-direction: column;
    min-width: 0;
    min-height: 0;
    position: relative;
    flex: 1;
    /* Fixed dark terminal background (matches XTERM_THEME.background); the
       `.term-out` hex, deliberately theme-independent. */
    background: #14161f;
  }
  .term-hd {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-bottom: 1px solid var(--bd0);
    flex-wrap: wrap;
    min-height: 44px;
    background: var(--bg1);
  }
  /* Rail-drawer toggle: hidden until the pane is too narrow to hold the rail
     inline (the query container is the view's `.term-shell`). */
  .termsb {
    display: none;
  }
  @container term-shell (max-width: 640px) {
    .termsb {
      display: inline-flex;
    }
  }
  /* Phone drilldown: the rail-toggle is meaningless (the rail is a separate screen);
     PhoneBack replaces it. A desktop narrow *pane* still gets the toggle above via
     the container query - only the phone viewport drops it (source order wins when
     both queries match). */
  @media (max-width: 640px) {
    .termsb {
      display: none;
    }
  }
  .cmdline {
    font: 600 var(--fs-md) var(--font-mono);
    color: var(--tx0);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    min-width: 0;
  }
  .grow {
    flex: 1;
    min-width: 0;
  }
  .term-out {
    flex: 1;
    overflow: hidden;
    background: #14161f;
    padding: 8px 6px 8px 12px;
    min-height: 0;
  }
  .term-out:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: -2px;
  }
  .term-jump {
    position: absolute;
    bottom: 44px;
    left: 50%;
    translate: -50% 0;
    z-index: 5;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--bg3);
    border: 1px solid var(--bd1);
    color: var(--tx0);
    font: 500 var(--fs-xs) var(--font-mono);
    padding: 5px 11px;
    border-radius: var(--r-full);
    box-shadow: var(--sh-2);
    cursor: pointer;
  }
  .armed-warn {
    position: absolute;
    bottom: 44px;
    right: 12px;
    z-index: 5;
    margin: 0;
    padding: 4px 9px;
    border-radius: var(--r-md);
    background: color-mix(in oklab, var(--st-err) 16%, transparent);
    border: 1px solid color-mix(in oklab, var(--st-err) 40%, transparent);
    color: var(--st-err);
    font: 500 var(--fs-2xs) var(--font-mono);
  }
  .term-status {
    display: flex;
    gap: 14px;
    align-items: center;
    padding: 5px 12px;
    border-top: 1px solid var(--bd0);
    background: var(--bg1);
    font: 500 var(--fs-2xs) var(--font-mono);
    color: var(--tx3);
    flex-wrap: wrap;
  }
  .term-status b {
    color: var(--tx2);
    font-weight: 600;
  }
  .term-status .cwd b {
    max-width: 220px;
    display: inline-block;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    vertical-align: bottom;
  }

  .t-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    height: 20px;
    padding: 0 7px;
    border-radius: var(--r-md);
    background: var(--bg2);
    border: 1px solid var(--bd0);
    font: 500 var(--fs-xs) var(--font-mono);
    color: var(--tx1);
    white-space: nowrap;
    max-width: 240px;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  /* The owning-session chip is a real link: chip skin on a button. */
  .t-chip.sess-link {
    cursor: pointer;
  }
  .t-chip.sess-link:hover {
    color: var(--acc);
    border-color: var(--bd1);
  }
  .t-chip :global(.ic) {
    width: 10px;
    height: 10px;
    color: var(--tx3);
  }

  /* Two-click kill: the armed confirm state the library Button doesn't model.
     .t-btn(+--sm/--danger/.is-armed) */
  .t-btn {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 5px;
    height: 23px;
    padding: 0 8px;
    border-radius: var(--r-md);
    border: 1px solid var(--bd1);
    font: 500 var(--fs-sm) / 1 var(--font-ui);
    cursor: pointer;
    white-space: nowrap;
    transition:
      background var(--t-1) var(--ease),
      border-color var(--t-1) var(--ease);
  }
  .t-btn--danger {
    background: color-mix(in oklab, var(--st-err) 13%, transparent);
    border-color: color-mix(in oklab, var(--st-err) 38%, transparent);
    color: var(--st-err);
  }
  .t-btn--danger:hover {
    background: color-mix(in oklab, var(--st-err) 22%, transparent);
    border-color: color-mix(in oklab, var(--st-err) 55%, transparent);
  }
  .t-btn--danger.is-armed {
    background: var(--st-err);
    border-color: var(--st-err);
    color: var(--bg0);
    --c: var(--st-err);
    animation: tpulse 1s var(--ease) infinite;
  }
  .t-btn--danger.is-armed :global(.ic) {
    color: var(--bg0);
  }
  .t-btn :global(.ic) {
    width: 13px;
    height: 13px;
  }
  .t-btn:focus-visible {
    outline: 2px solid var(--acc);
    outline-offset: 1px;
  }
  @keyframes tpulse {
    0%,
    100% {
      box-shadow: 0 0 0 0 color-mix(in oklab, var(--c, var(--acc)) 45%, transparent);
    }
    55% {
      box-shadow: 0 0 0 5px transparent;
    }
  }
  @media (prefers-reduced-motion: reduce) {
    .t-btn--danger.is-armed {
      animation: none;
    }
  }
</style>
