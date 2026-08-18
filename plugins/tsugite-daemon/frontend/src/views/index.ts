import type { Component } from 'svelte';
import type { IconName } from '$lib/components/icon/icons';
import { pluginsMeta, type PluginSurface } from '$lib/stores/pluginsMeta.svelte';

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

const builtinViews: ViewDef[] = import.meta.env.DEV ? [...views, galleryView] : views;

/** Rail order: built-ins first, then whatever plugins contributed this session.
 *  A function, because the plugin entries arrive after boot and callers read it
 *  inside reactive scopes. A plugin row's id doubles as its surface kind, which
 *  is what makes #plugin/<name>/<kind> a deep link. */
export function allViews(): ViewDef[] {
  const contributed = pluginsMeta.surfaces
    .filter((surface) => surface.nav)
    .map((surface): ViewDef => {
      const row = {
        id: surface.kind,
        label: surface.label,
        icon: surface.icon,
        mode: surface.mode,
      };
      if (surface.mode === 'workspace') return row;
      return { ...row, load: () => import('$lib/components/plugins/PluginView.svelte') };
    });
  return [...builtinViews, ...contributed];
}

/** The surface a nav view opens into the mux instead of replacing the region, or
 *  null when the view owns the whole region. Only a plugin surface that declared
 *  workspace mode docks; a built-in workspace view opens its own tabs from the rail. */
export function dockedSurface(viewId: string): PluginSurface | null {
  const surface = pluginsMeta.byKind(viewId);
  return surface?.mode === 'workspace' ? surface : null;
}

export function viewById(id: string): ViewDef {
  // An unknown id resolves to the default view.
  return allViews().find((view) => view.id === id) ?? views[0]!;
}
