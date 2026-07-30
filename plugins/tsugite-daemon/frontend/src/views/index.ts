import type { Component } from 'svelte';
import type { IconName } from '$lib/components/icon/icons';

/**
 * A workspace view (chats/terminals/files) drives the shared context rail + the
 * one mux tab area; it renders no full-area component of its own (its content is
 * the docked surfaces). A full view replaces the whole workspace region with the
 * component its `load` resolves - lazily, so each full view is its own JS chunk and
 * a user only downloads the views they actually open.
 */
export type ViewMode = 'workspace' | 'full';

export interface ViewDef {
  id: string;
  label: string;
  /** Full views lazy-load their component here; workspace views omit it. */
  load?: () => Promise<{ default: Component }>;
  /** Nav-rail glyph. */
  icon: IconName;
  mode: ViewMode;
}

// Do not reorder without reason.
export const views: ViewDef[] = [
  { id: 'chats', label: 'Chats', icon: 'chat', mode: 'workspace' },
  { id: 'terminals', label: 'Terminals', icon: 'term', mode: 'workspace' },
  { id: 'files', label: 'Files', icon: 'files', mode: 'workspace' },
  {
    id: 'jobs',
    label: 'Jobs',
    load: () => import('./jobs/View.svelte'),
    icon: 'jobs',
    mode: 'full',
  },
  {
    id: 'schedules',
    label: 'Schedules',
    load: () => import('./schedules/View.svelte'),
    icon: 'sched',
    mode: 'full',
  },
  {
    id: 'usage',
    label: 'Usage',
    load: () => import('./usage/View.svelte'),
    icon: 'usage',
    mode: 'full',
  },
  {
    id: 'agents',
    label: 'Agents',
    load: () => import('./agents/View.svelte'),
    icon: 'agent',
    mode: 'full',
  },
  {
    id: 'skills',
    label: 'Skills',
    load: () => import('./skills/View.svelte'),
    icon: 'skill',
    mode: 'full',
  },
  {
    id: 'tools',
    label: 'Tools',
    load: () => import('./tools/View.svelte'),
    icon: 'tool',
    mode: 'full',
  },
  {
    id: 'webhooks',
    label: 'Webhooks',
    load: () => import('./webhooks/View.svelte'),
    icon: 'hook',
    mode: 'full',
  },
  {
    id: 'hooks',
    label: 'Hooks',
    load: () => import('./hooks/View.svelte'),
    icon: 'fork',
    mode: 'full',
  },
  {
    id: 'secrets',
    label: 'Secrets',
    load: () => import('./secrets/View.svelte'),
    icon: 'key',
    mode: 'full',
  },
  {
    id: 'plugins',
    label: 'Plugins',
    load: () => import('./plugins/View.svelte'),
    icon: 'plug',
    mode: 'full',
  },
];

// Dev-only surface that auto-discovers *.gallery.svelte demos. Reachable at
// #gallery and shown in the nav rail in dev builds only - unreachable in prod.
export const galleryView: ViewDef = {
  id: 'gallery',
  label: 'Gallery',
  load: () => import('./gallery/View.svelte'),
  icon: 'grid',
  mode: 'full',
};

export const allViews: ViewDef[] = import.meta.env.DEV ? [...views, galleryView] : views;

// The shell's default surface: the first rail entry. Single source of truth for
// the fallback view - App's empty-pane rail highlight and viewById's unknown-id
// fallback both resolve here, and spaces' default 'chat' surface aliases to it.
export const DEFAULT_VIEW_ID = views[0]!.id;

export function viewById(id: string): ViewDef {
  // An unknown id resolves to the default view. views is a non-empty literal and
  // DEFAULT_VIEW_ID is views[0].id, so this lookup is always defined.
  return allViews.find((view) => view.id === id) ?? allViews.find((v) => v.id === DEFAULT_VIEW_ID)!;
}
