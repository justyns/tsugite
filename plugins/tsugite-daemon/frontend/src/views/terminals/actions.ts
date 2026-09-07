import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { terminals, type Terminal } from '$lib/stores/terminals.svelte';

export async function killTerminal(term: Terminal): Promise<void> {
  try {
    await terminals.kill(term.id);
    toasts.push('warn', 'Terminal killed', { body: `${term.cmd.slice(0, 44)} · record kept` });
  } catch (err) {
    toasts.push('err', 'Kill failed', { body: err instanceof Error ? err.message : String(err) });
  }
}

export async function restartTerminal(term: Terminal): Promise<Terminal | null> {
  try {
    const next = await terminals.restart(term.id);
    toasts.push('ok', 'PTY restarted', { body: `${next.id} · restarted from ${term.id}` });
    return next;
  } catch (err) {
    toasts.push('err', 'Restart failed', {
      body: err instanceof Error ? err.message : String(err),
    });
    return null;
  }
}
