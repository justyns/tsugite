// Pure display helpers for the plugin registry table.
import type { PluginInfo } from '$lib/stores/pluginsMeta.svelte';

const GROUP_PREFIX = 'tsugite.';

/** Drops the entry-point-group prefix (mirrors `tsu plugins list`'s own
 *  `group.removeprefix("tsugite.")`), e.g. "tsugite.adapters" -> "adapters". */
export function groupLabel(group: string): string {
  return group.startsWith(GROUP_PREFIX) ? group.slice(GROUP_PREFIX.length) : group;
}

/** discover_plugins() scans 11 groups independently, so the same distribution
 *  can register the same entry-point name in more than one group (e.g. an
 *  adapter and a tool both named "cc_driver") - group+name is the real unique
 *  key, name alone is not. */
export function pluginKey(p: Pick<PluginInfo, 'group' | 'name'>): string {
  return `${p.group}:${p.name}`;
}

/** `importlib.metadata.entry_points()` makes no ordering guarantee, so the
 *  table sorts client-side for a stable, predictable row order (group, then
 *  name) - the same key `tsu plugins list` sorts by. */
export function sortPlugins(plugins: PluginInfo[]): PluginInfo[] {
  return [...plugins].sort(
    (a, b) => a.group.localeCompare(b.group) || a.name.localeCompare(b.name),
  );
}
