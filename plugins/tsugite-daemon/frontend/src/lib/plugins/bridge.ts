/**
 * Host↔plugin postMessage protocol for plugin UI surfaces, versioned from day one.
 *
 * The host posts `init` once the iframe loads and the plugin answers `ready`; a
 * plugin that never answers gets an error state rather than a blank pane. `init`
 * and `theme` carry the resolved design-token values so a plugin page can skin
 * itself across all five themes without importing anything from the host.
 *
 * Pure parse/build so the protocol is unit-testable; the DOM wiring lives in
 * PluginSurface.svelte.
 */
import type { PluginSurface } from '$lib/stores/pluginsMeta.svelte';

export const BRIDGE_VERSION = 1;

/** How long the host waits for `ready` before showing the error state. */
export const READY_TIMEOUT_MS = 10_000;

/** A plugin-set tab title longer than this would push the tab strip around. */
const MAX_TITLE = 120;

export interface ThemePayload {
  name: string;
  tokens: Record<string, string>;
}

export interface InitMessage {
  type: 'tsugite:init';
  version: number;
  surface: { kind: string; params: Record<string, string> };
  theme: ThemePayload;
}

export interface ThemeMessage {
  type: 'tsugite:theme';
  theme: ThemePayload;
}

export type ReadyMessage = { type: 'tsugite:ready' };
export type TitleMessage = { type: 'tsugite:title'; title: string };
export type PluginMessage = ReadyMessage | TitleMessage;

/** Resolved values of the active theme's design tokens, read off the element the
 *  theme is applied to (<html>). Enumerated rather than listed, so a token added
 *  to tokens.css reaches plugins without a second edit here. */
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
): InitMessage {
  // Copied, not passed through: params arrive as a tab's reactive state, and
  // postMessage's structured clone rejects a proxy outright.
  return {
    type: 'tsugite:init',
    version: BRIDGE_VERSION,
    surface: { kind, params: { ...params } },
    theme,
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
  if (type === 'tsugite:title' && typeof title === 'string') {
    return { type, title: title.trim().slice(0, MAX_TITLE) };
  }
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
