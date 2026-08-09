/**
 * Maps a docked surface kind to the component that renders its content inside a
 * mux tab. Kept apart from `views/index.ts` (which is the nav registry) and from
 * the pure `shellNav.ts` vocabulary helpers so those stay component-free.
 *
 * Built-ins are compiled in; every plugin surface renders through the one generic
 * PluginSurface, so a plugin ships HTML rather than a Svelte component. The
 * daemon owns the `plugin/` namespace, so the prefix is enough to route a kind
 * there - including a persisted tab whose plugin is now uninstalled, which
 * PluginSurface explains rather than dropping.
 */
import type { Component } from 'svelte';
import PluginSurface from '$lib/components/plugins/PluginSurface.svelte';
import ChatSurface from './chats/Surface.svelte';
import TerminalSurface from './terminals/Surface.svelte';
import FileSurface from './files/Surface.svelte';

export type SurfaceProps = {
  params?: Record<string, string>;
  /** The tab's surface kind; plugin surfaces resolve their entry URL from it. */
  kind?: string;
  /** Rename the tab this surface is mounted in. */
  setTitle?: (title: string) => void;
};

const SURFACES: Record<string, Component<SurfaceProps>> = {
  chat: ChatSurface as Component<SurfaceProps>,
  terminal: TerminalSurface as Component<SurfaceProps>,
  file: FileSurface as Component<SurfaceProps>,
};

export function surfaceComponent(kind: string): Component<SurfaceProps> | undefined {
  return (
    SURFACES[kind] ??
    (kind.startsWith('plugin/') ? (PluginSurface as Component<SurfaceProps>) : undefined)
  );
}
