/**
 * Host↔plugin postMessage protocol for plugin UI surfaces. See the bridge
 * section of docs/plugin-adapters.md for the wire contract.
 */
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
}

interface ThemeMessage {
  type: 'tsugite:theme';
  theme: ThemePayload;
}

type PluginMessage = { type: 'tsugite:ready' } | { type: 'tsugite:title'; title: string };

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
): InitMessage {
  // Copied, not passed through: params arrive as a tab's reactive state, and
  // postMessage's structured clone rejects a proxy outright.
  return {
    type: 'tsugite:init',
    version: BRIDGE_VERSION,
    surface: { kind, params: { ...params } },
    theme,
    token,
  };
}

export function themeMessage(theme: ThemePayload): ThemeMessage {
  return { type: 'tsugite:theme', theme };
}

/** A message from the iframe, or null for anything this version doesn't speak.
 *  The caller has already checked the sender is the surface's own frame. */
export function parsePluginMessage(data: unknown): PluginMessage | null {
  if (typeof data !== 'object' || data === null) return null;
  const { type, title } = data as { type?: unknown; title?: unknown };
  if (type === 'tsugite:ready') return { type };
  if (type === 'tsugite:title' && typeof title === 'string') return { type, title: title.trim() };
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
