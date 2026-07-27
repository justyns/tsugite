/**
 * Attach a session / job / terminal record to a chat as context by an explicit
 * action, replacing the old regex-on-paste detector. Two entry points:
 *   - attachRecordToChat: resolve a target chat, capture the record on the daemon,
 *     and push it into that chat's composer (via the contextAttach signal).
 *   - copyReference: write a clipboard marker a composer paste attaches (its
 *     read/parse half is parseRefMarker, kept here so the format round-trips).
 */
import { sessions, type SessionRow } from '$lib/stores/sessions.svelte';
import { router, navigate } from '$lib/router.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { captureServerContext } from '$lib/context/serverProviders';
import { isFinishedSession } from './sessionModel';
import { contextAttach } from './contextAttach.svelte';

export type RecordKind = 'session' | 'job' | 'terminal';

/** The chat a captured record attaches to: the focused chat if one is open, else
 *  the primary session, else the most-recent still-open chat. Null when there is
 *  no chat to attach to. A focused chat the sessions store hasn't loaded a row for
 *  (a deep-linked other-agent chat) still resolves by its id. */
function resolveTargetChat(): { id: string; title: string } | null {
  const focused = router.view === 'chats' ? (router.params.sessionId ?? null) : null;
  const rows = sessions.ordered;
  const title = (row: SessionRow | undefined) => row?.title || row?.label || 'chat';
  // A focused chat wins even when the store hasn't loaded a row for it (a
  // deep-linked other-agent chat); the title just falls back to a generic one.
  if (focused) return { id: focused, title: title(rows.find((r) => r.id === focused)) };
  const row = rows.find((r) => r.is_primary) ?? rows.find((r) => !isFinishedSession(r));
  return row ? { id: row.id, title: title(row) } : null;
}

/** Capture a record on the daemon and push it into the target chat's composer as
 *  a context chip, then focus that chat so the chip is visible. */
export async function attachRecordToChat(kind: RecordKind, id: string): Promise<void> {
  const target = resolveTargetChat();
  if (!target) {
    toasts.push('warn', 'Open a chat first');
    return;
  }
  let items;
  try {
    items = await captureServerContext(kind, target.id, id);
  } catch (err) {
    toasts.push('err', 'Could not attach', {
      body: err instanceof Error ? err.message : String(err),
    });
    return;
  }
  if (!items.length) {
    toasts.push('warn', 'Nothing to attach');
    return;
  }
  contextAttach.request(target.id, items);
  toasts.push('ok', `Added to ${target.title}`);
  navigate('chats', { sessionId: target.id });
}

// A reference marker: kind:id lives in a data attribute so the html round-trips
// as an attachable token, while the text twin stays a meaningful human string
// when pasted anywhere that isn't a tsugite composer.
function refText(kind: RecordKind, id: string): string {
  return `${kind} ${id}`;
}
export function refMarkerHtml(kind: RecordKind, id: string): string {
  return `<span data-tsugite-ref="${kind}:${id}">${refText(kind, id)}</span>`;
}

/** Read a reference marker out of pasted html; null when it isn't one. */
export function parseRefMarker(html: string): { kind: string; id: string } | null {
  const m = /data-tsugite-ref="([a-z]+):([^"]+)"/.exec(html);
  return m && m[1] && m[2] ? { kind: m[1], id: m[2] } : null;
}

// Fallback when ClipboardItem / clipboard.write is unavailable: copy the human
// string via a detached textarea. The html attach-marker is dropped in this path
// (a textarea is plain-text only), so pasting elsewhere still reads sensibly.
function copyTextFallback(text: string): void {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
  } finally {
    ta.remove();
  }
}

/** Copy a reference to a record: text/plain (a human string, so pasting elsewhere
 *  is meaningful) plus text/html (the attach marker, so pasting into a composer
 *  attaches it). */
export async function copyReference(kind: RecordKind, id: string): Promise<void> {
  const text = refText(kind, id);
  const html = refMarkerHtml(kind, id);
  try {
    if (typeof ClipboardItem !== 'undefined' && navigator.clipboard?.write) {
      await navigator.clipboard.write([
        new ClipboardItem({
          'text/plain': new Blob([text], { type: 'text/plain' }),
          'text/html': new Blob([html], { type: 'text/html' }),
        }),
      ]);
    } else {
      copyTextFallback(text);
    }
    toasts.push('ok', 'Reference copied', { body: text });
  } catch {
    toasts.push('err', 'Could not copy reference');
  }
}
