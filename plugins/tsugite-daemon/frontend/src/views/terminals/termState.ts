/**
 * Pure state/format mappings for the terminals view (kept out of the .svelte
 * files so they're directly node-unit-testable). The 6 backend TerminalState
 * values are folded onto the pill vocabulary + rail indicator glyphs.
 * "paused-follow" is deliberately absent - it is a client boolean on the
 * store, not a server state.
 */
import type { IconName } from '$lib/components/icon/icons';
import type { TerminalState } from '$lib/stores/terminals.svelte';
import type { PaneTabState } from '$lib/shell/mux/layout';

/** Header status pill: color + icon + text together (never color alone). `st`
 *  is the `data-st` bucket (reuses the job/state-language pill colors);
 *  `spin` swaps the icon for the braille activity spinner. */
export interface TermPillSpec {
  st: 'queued' | 'running' | 'done' | 'errored' | 'cancelled' | 'stuck';
  label: string;
  spin: boolean;
  icon: IconName;
}

export function terminalPill(state: TerminalState, exitCode?: number | null): TermPillSpec {
  switch (state) {
    case 'starting':
      return { st: 'queued', label: 'starting', spin: true, icon: 'clock' };
    case 'running':
      return { st: 'running', label: 'running', spin: true, icon: 'play' };
    case 'succeeded':
      return { st: 'done', label: exitLabel(exitCode, 0), spin: false, icon: 'check' };
    case 'failed':
      return { st: 'errored', label: exitLabel(exitCode, 1), spin: false, icon: 'x' };
    case 'cancelled':
      return { st: 'cancelled', label: 'killed', spin: false, icon: 'cancel' };
    case 'stream_lost':
      return { st: 'stuck', label: 'stream lost', spin: false, icon: 'alert' };
  }
}

function exitLabel(exitCode: number | null | undefined, fallback: number): string {
  const code = exitCode == null ? fallback : exitCode;
  return `exit ${code}`;
}

/** Rail-row leading glyph. `spin` renders the braille spinner, else `icon`;
 *  each state's tint is a stylesheet rule on the row's data-st (never inline). */
export interface TermIndicatorSpec {
  spin: boolean;
  icon: IconName;
}

export function terminalIndicator(state: TerminalState): TermIndicatorSpec {
  switch (state) {
    case 'running':
      return { spin: true, icon: 'play' };
    case 'starting':
      return { spin: true, icon: 'clock' };
    case 'stream_lost':
      return { spin: false, icon: 'alert' };
    case 'failed':
      return { spin: false, icon: 'x' };
    case 'cancelled':
      return { spin: false, icon: 'cancel' };
    case 'succeeded':
      return { spin: false, icon: 'check' };
  }
}

/** The dot state a docked terminal tab shows in the mux tab-strip. */
export function terminalTabState(state: TerminalState): PaneTabState {
  switch (state) {
    case 'running':
    case 'starting':
      return 'busy';
    case 'succeeded':
      return 'done';
    case 'failed':
      return 'error';
    case 'stream_lost':
      return 'blocked';
    case 'cancelled':
      return 'idle';
  }
}

/** Whether the terminal is live (accepts stdin / can be killed). */
export function isLiveTerminal(state: TerminalState): boolean {
  return state === 'starting' || state === 'running';
}

/** Byte count -> "915 B" / "48 KB" / "2.1 MB". */
export function formatBytes(n: number): string {
  if (n > 1e6) return `${(n / 1e6).toFixed(1)} MB`;
  if (n > 1000) return `${Math.round(n / 1000)} KB`;
  return `${n} B`;
}

/** Seconds a terminal has been alive: created -> (resolved || now), never
 *  negative. Bad/missing timestamps collapse to 0. Compose with
 *  feedback/format.ts formatElapsed() for the mm:ss label. */
export function elapsedSeconds(
  createdAt: string,
  resolvedAt: string | null,
  nowMs: number,
): number {
  const start = Date.parse(createdAt);
  if (Number.isNaN(start)) return 0;
  const end = resolvedAt ? Date.parse(resolvedAt) : nowMs;
  if (Number.isNaN(end)) return 0;
  return Math.max(0, (end - start) / 1000);
}
