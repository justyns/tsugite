/**
 * Maps a docked surface kind to the component that renders its content inside a
 * mux tab. Kept apart from `views/index.ts` (which is the nav registry) and from
 * the pure `shellNav.ts` vocabulary helpers so those stay component-free.
 */
import type { Component } from 'svelte';
import ChatSurface from './chats/Surface.svelte';
import TerminalSurface from './terminals/Surface.svelte';
import FileSurface from './files/Surface.svelte';

export type SurfaceProps = { params?: Record<string, string> };

const SURFACES: Record<string, Component<SurfaceProps>> = {
  chat: ChatSurface as Component<SurfaceProps>,
  terminal: TerminalSurface as Component<SurfaceProps>,
  file: FileSurface as Component<SurfaceProps>,
};

export function surfaceComponent(kind: string): Component<SurfaceProps> | undefined {
  return SURFACES[kind];
}
