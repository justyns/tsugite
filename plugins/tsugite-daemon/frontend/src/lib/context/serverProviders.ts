/**
 * Server-context providers: the daemon-captured half of the composer's "add
 * context" menu. A plugin registers a provider on the daemon; this module lists
 * them (GET /api/context-providers) and runs a pick server-side - a plain capture
 * (POST .../capture), or, when the provider offers a submenu, its choices
 * (GET .../choices) followed by a capture with the chosen value as the `arg`.
 *
 * Client providers (contextProviders.ts) capture in the browser; these capture on
 * the daemon. Both hand back the same {key,label,value} items that ride a send as
 * context_metadata, so the chip + gutter render is shared.
 */
import { api } from '$lib/api/client';
import { ICONS, type IconName } from '$lib/components/icon/icons';
import type { ContextChoice } from '$lib/components/composer/types';
import type { ContextItem } from './contextProviders';

/** A daemon-provided context provider, with the wire fields normalized. */
export interface ServerProvider {
  key: string;
  label: string;
  icon: IconName;
  hasChoices: boolean;
  /** Its option set is large/searchable: open the Picker overlay, not the inline submenu. */
  picker: boolean;
  /** Shows in the composer's add-context menu (an autocomplete-only source is
   *  listed here but rides in_menu=false so the menu excludes it). */
  inMenu: boolean;
  /** When set, this is an `@<prefix> <query>` autocomplete source. */
  autocompletePrefix: string | null;
}

interface ProviderWire {
  key: string;
  label: string;
  icon: string;
  has_choices: boolean;
  picker?: boolean;
  in_menu?: boolean;
  autocomplete_prefix?: string | null;
}

/** Fall back to a generic glyph for an icon a third-party plugin names that this
 *  build doesn't ship (Icon reads ICONS[name] and would throw on an unknown one). */
function toIcon(name: string): IconName {
  return name in ICONS ? (name as IconName) : 'sparkle';
}

/** List the daemon's menu providers. Best-effort: an unreachable daemon or one
 *  without the feature leaves the menu with only its client providers. */
export async function fetchServerProviders(): Promise<ServerProvider[]> {
  try {
    const res = await api.get<{ providers: ProviderWire[] }>('/api/context-providers');
    return (res.providers ?? []).map((p) => ({
      key: p.key,
      label: p.label,
      icon: toIcon(p.icon),
      hasChoices: Boolean(p.has_choices),
      picker: Boolean(p.picker),
      // A daemon that predates the field (or a test stub omitting it) returned only
      // menu providers, so a missing flag means "in the menu".
      inMenu: p.in_menu ?? true,
      autocompletePrefix: p.autocomplete_prefix ?? null,
    }));
  } catch {
    return [];
  }
}

/** The submenu options for a provider that offers them, scoped to the active
 *  session. Best-effort: an error yields no options (the submenu simply won't open). */
export async function fetchServerChoices(key: string, sessionId: string): Promise<ContextChoice[]> {
  try {
    const res = await api.get<{ choices: ContextChoice[] }>(
      `/api/context-providers/${encodeURIComponent(key)}/choices?session_id=${encodeURIComponent(sessionId)}`,
    );
    return res.choices ?? [];
  } catch {
    return [];
  }
}

/** Query an autocomplete source as the user types, scoped to the active session.
 *  Best-effort: an error yields no results, so a flaky source never breaks the
 *  composer (the popover just shows empty). */
export async function searchServerProvider(
  key: string,
  sessionId: string,
  q: string,
): Promise<ContextChoice[]> {
  try {
    const res = await api.get<{ results: ContextChoice[] }>(
      `/api/context-providers/${encodeURIComponent(key)}/search?session_id=${encodeURIComponent(sessionId)}&q=${encodeURIComponent(q)}`,
    );
    return res.results ?? [];
  } catch {
    return [];
  }
}

/** Capture a provider's items on the daemon. `arg` is a chosen submenu value, or
 *  null for a provider with no choices. Rejects on a provider error the daemon
 *  returns as 400 - the pick was deliberate, so the caller surfaces it. */
export async function captureServerContext(
  key: string,
  sessionId: string,
  arg: string | null,
): Promise<ContextItem[]> {
  const res = await api.post<{ items: ContextItem[] }>(
    `/api/context-providers/${encodeURIComponent(key)}/capture`,
    { session_id: sessionId, arg },
  );
  return res.items ?? [];
}
