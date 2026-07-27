/**
 * Per-session composer draft persistence: the typed text plus anything staged
 * but not yet sent (uploaded attachments and attached context items).
 *
 * The text reuses the Alpine wire key scheme (tsugite_draft_${sessionId||'new'})
 * so an old-UI draft survives the rebuild. The staged items ride a companion key
 * so a phone that sleeps mid-compose (the PWA reloads on resume) restores its
 * attached photos and context chips, not just the words - both were already
 * uploaded/captured, so only their references are stored. localStorage-guarded
 * via $lib/storage.
 */
import { readLocal, removeLocal, writeLocal } from '$lib/storage';
import type { Attachment } from '$lib/components/composer/types';
import type { ContextItem } from '$lib/context/contextProviders';

export interface DraftStaged {
  attachments: Attachment[];
  contextItems: ContextItem[];
}

function draftKey(sessionId: string | null): string {
  return `tsugite_draft_${sessionId ?? 'new'}`;
}

function stagedKey(sessionId: string | null): string {
  return `tsugite_draft_staged_${sessionId ?? 'new'}`;
}

export function readDraft(sessionId: string | null): string {
  return readLocal(draftKey(sessionId)) ?? '';
}

export function writeDraft(sessionId: string | null, text: string): void {
  if (text) writeLocal(draftKey(sessionId), text);
  else removeLocal(draftKey(sessionId));
}

/** Staged attachments + context items for the session, or empty when none or
 *  corrupt. */
export function readDraftStaged(sessionId: string | null): DraftStaged {
  const raw = readLocal(stagedKey(sessionId));
  if (!raw) return { attachments: [], contextItems: [] };
  try {
    const p = (JSON.parse(raw) ?? {}) as Partial<DraftStaged>;
    return {
      attachments: Array.isArray(p.attachments) ? p.attachments : [],
      contextItems: Array.isArray(p.contextItems) ? p.contextItems : [],
    };
  } catch {
    return { attachments: [], contextItems: [] };
  }
}

export function writeDraftStaged(sessionId: string | null, staged: DraftStaged): void {
  if (staged.attachments.length || staged.contextItems.length) {
    writeLocal(stagedKey(sessionId), JSON.stringify(staged));
  } else {
    removeLocal(stagedKey(sessionId));
  }
}

export function clearDraft(sessionId: string | null): void {
  removeLocal(draftKey(sessionId));
  removeLocal(stagedKey(sessionId));
}
