import type { Component } from 'svelte';
import type { IconName } from '$lib/components/icon/icons';
import JobsView from './jobs/View.svelte';
import SchedulesView from './schedules/View.svelte';
import AgentsView from './agents/View.svelte';
import SkillsView from './skills/View.svelte';
import PluginsView from './plugins/View.svelte';
import SecretsView from './secrets/View.svelte';
import UsageView from './usage/View.svelte';
import WebhooksView from './webhooks/View.svelte';
import HooksView from './hooks/View.svelte';
import ToolsView from './tools/View.svelte';
import GalleryView from './gallery/View.svelte';

/**
 * A workspace view (chats/terminals/files) drives the shared context rail + the
 * one mux tab area; it renders no full-area component of its own (its content is
 * the docked surfaces). A full view replaces the whole workspace region with its
 * `component`.
 */
export type ViewMode = 'workspace' | 'full';

export interface ViewDef {
  id: string;
  label: string;
  /** Full views render this in the workspace region; workspace views omit it. */
  component?: Component;
  /** Nav-rail glyph. */
  icon: IconName;
  mode: ViewMode;
}

// Do not reorder without reason.
export const views: ViewDef[] = [
  { id: 'chats', label: 'Chats', icon: 'chat', mode: 'workspace' },
  { id: 'terminals', label: 'Terminals', icon: 'term', mode: 'workspace' },
  { id: 'files', label: 'Files', icon: 'files', mode: 'workspace' },
  { id: 'jobs', label: 'Jobs', component: JobsView, icon: 'jobs', mode: 'full' },
  { id: 'schedules', label: 'Schedules', component: SchedulesView, icon: 'sched', mode: 'full' },
  { id: 'usage', label: 'Usage', component: UsageView, icon: 'usage', mode: 'full' },
  { id: 'agents', label: 'Agents', component: AgentsView, icon: 'agent', mode: 'full' },
  { id: 'skills', label: 'Skills', component: SkillsView, icon: 'skill', mode: 'full' },
  { id: 'tools', label: 'Tools', component: ToolsView, icon: 'tool', mode: 'full' },
  { id: 'webhooks', label: 'Webhooks', component: WebhooksView, icon: 'hook', mode: 'full' },
  { id: 'hooks', label: 'Hooks', component: HooksView, icon: 'fork', mode: 'full' },
  { id: 'secrets', label: 'Secrets', component: SecretsView, icon: 'key', mode: 'full' },
  { id: 'plugins', label: 'Plugins', component: PluginsView, icon: 'plug', mode: 'full' },
];

// Dev-only surface that auto-discovers *.gallery.svelte demos. Reachable at
// #gallery and shown in the nav rail in dev builds only - unreachable in prod.
export const galleryView: ViewDef = {
  id: 'gallery',
  label: 'Gallery',
  component: GalleryView,
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
