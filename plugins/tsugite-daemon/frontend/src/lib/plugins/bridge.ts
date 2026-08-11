/**
 * Host↔plugin postMessage protocol for plugin UI surfaces. See the bridge
 * section of docs/plugin-adapters.md for the wire contract.
 */
import type { SSEEvent } from '$lib/api/sse';
import type { PluginSurface } from '$lib/stores/pluginsMeta.svelte';

export const BRIDGE_VERSION = 1;

/** How long the host waits for `ready` before showing the error state. */
export const READY_TIMEOUT_MS = 10_000;

export interface ThemePayload {
  name: string;
  tokens: Record<string, string>;
}

interface InitMessage {
  type: 'tsugite:init';
  version: number;
  surface: { kind: string; params: Record<string, string> };
  theme: ThemePayload;
  /** The daemon bearer token, for calling the plugin's own authed routes. */
  token: string;
  /** Who is viewing, so a surface can attribute what the human does in it apart
   *  from what an agent does through the plugin's tools. */
  user: string;
}

interface ThemeMessage {
  type: 'tsugite:theme';
  theme: ThemePayload;
}

interface EventMessage {
  type: 'tsugite:event';
  event: { type: string; data: Record<string, unknown> };
}

type PluginMessage =
  { type: 'tsugite:ready' } | { type: 'tsugite:title'; title: string } | { type: 'tsugite:focus' };

/** Enumerated, so a token added to tokens.css reaches plugins with no edit here. */
export function readThemeTokens(el: Element): Record<string, string> {
  const style = getComputedStyle(el);
  const tokens: Record<string, string> = {};
  for (const name of style) {
    if (name.startsWith('--')) tokens[name] = style.getPropertyValue(name).trim();
  }
  return tokens;
}

export function initMessage(
  kind: string,
  params: Record<string, string>,
  theme: ThemePayload,
  token: string,
  user: string,
): InitMessage {
  // Copied, not passed through: params arrive as a tab's reactive state, and
  // postMessage's structured clone rejects a proxy outright.
  return {
    type: 'tsugite:init',
    version: BRIDGE_VERSION,
    surface: { kind, params: { ...params } },
    theme,
    token,
    user,
  };
}

export function themeMessage(theme: ThemePayload): ThemeMessage {
  return { type: 'tsugite:theme', theme };
}

/** One daemon broadcast frame, on its way into a surface that declared its type.
 *  `seq` stays behind: the host owns the stream's cursor, and a page that never
 *  reconnects has nothing to do with one. */
export function eventMessage(event: SSEEvent): EventMessage {
  return { type: 'tsugite:event', event: { type: event.type, data: event.data ?? {} } };
}

/** A message from the iframe, or null for anything this version doesn't speak.
 *  The caller has already checked the sender is the surface's own frame. */
export function parsePluginMessage(data: unknown): PluginMessage | null {
  if (typeof data !== 'object' || data === null) return null;
  const { type, title } = data as { type?: unknown; title?: unknown };
  if (type === 'tsugite:ready') return { type };
  if (type === 'tsugite:title' && typeof title === 'string') return { type, title: title.trim() };
  if (type === 'tsugite:focus') return { type };
  return null;
}

/** The iframe URL for a surface: its declared entry plus only the params the
 *  surface asked for, so an unrelated tab param never leaks into a plugin. */
export function surfaceSrc(surface: PluginSurface, params: Record<string, string>): string {
  const query = new URLSearchParams();
  for (const name of surface.params) {
    const value = params[name];
    if (value !== undefined) query.set(name, value);
  }
  const q = query.toString();
  return q ? `${surface.entry}?${q}` : surface.entry;
}
