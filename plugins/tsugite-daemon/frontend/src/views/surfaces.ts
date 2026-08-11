/**
 * Maps a docked surface kind to the component that renders its content inside a
 * mux tab. Kept apart from `views/index.ts` (which is the nav registry) and from
 * the pure `shellNav.ts` vocabulary helpers so those stay component-free.
 *
 * Built-ins are compiled in; every plugin surface renders through the one generic
 * PluginSurface, so a plugin ships HTML and never a Svelte component.
 */
import type { Component } from 'svelte';
import PluginSurface from '$lib/components/plugins/PluginSurface.svelte';
import ChatSurface from './chats/Surface.svelte';
import TerminalSurface from './terminals/Surface.svelte';
import FileSurface from './files/Surface.svelte';

export type SurfaceProps = {
  params?: Record<string, string>;
  kind?: string;
  /** Rename the tab this surface is mounted in; only the host that owns a tab
   *  supplies it, so a surface can never address another tab. */
  setTitle?: (title: string) => void;
  /** Make this surface's pane the focused one. Absent in a full view, which has
   *  no pane to claim. */
  focusPane?: () => void;
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
