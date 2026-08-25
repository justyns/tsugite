/**
 * Client-context controller for the composer: the manually-attached chips
 * (`contextItems`), the "add context" menu (client registry + daemon providers),
 * the @ reference popover's candidates/search sources, and the picker overlay.
 * Capturing an item (menu pick, @ pick, reference paste, or add-to-chat) chips it;
 * a send folds these plus any auto-attach providers into `context_metadata`.
 *
 * A mutated $state class instance, never a reassigned binding (AGENTS.md): the
 * component instantiates it, wires the effects that load its provider/file lists,
 * and reads its derived menus + chips in the markup.
 */
import type {
  ContextMenuItem,
  ContextChoice,
  RefItem,
  RefSource,
} from '$lib/components/composer/types';
import { contextProviders, contextProvider, type ContextItem } from '$lib/context/contextProviders';
import {
  fetchServerProviders,
  fetchServerChoices,
  captureServerContext,
  searchServerProvider,
  type ServerProvider,
} from '$lib/context/serverProviders';
import { sessions } from '$lib/stores/sessions.svelte';
import { buildSessionRefs } from './sessionRefs';
import { autoAttachStore } from '$lib/stores/autoAttach.svelte';
import { toasts } from '$lib/components/feedback/toast-store.svelte';
import { loadWorkspace } from '../files/load';

// Auto-attach capture failures notify once per provider per app session, not on
// every send, so a denied permission with the setting on doesn't spam a toast
// per message.
const autoAttachWarned = new Set<string>();

export interface ContextDeps {
  readonly sessionId: string | null;
}

export class ContextItems {
  #deps: ContextDeps;
  #refsLoaded = false;

  // Manually-attached context items (chips). Provider key -> item.
  contextItems = $state<ContextItem[]>([]);
  // Daemon-provided menu providers, loaded once (best-effort - empty if the daemon
  // lacks the feature). Client providers below always show regardless.
  serverProviders = $state<ServerProvider[]>([]);
  // Workspace files feeding the @ popover, loaded once (best-effort).
  // Picking a file attaches it as a workspace-file context item rather than
  // inserting `@path` text (see pickRef).
  fileRefs = $state<RefItem[]>([]);
  // The generic Picker overlay, opened when a large-option (picker) provider is
  // chosen; it lists that provider's choices and captures the pick as a chip.
  picker = $state<{ item: ContextMenuItem; items: ContextChoice[] } | null>(null);

  // The @ popover's built-in candidates: recent sessions first, then workspace
  // files (both grouped). Reactive to the sessions store and the open chat, so
  // the source stays current and never lists the chat you are in.
  readonly refItems: RefItem[] = $derived.by(() => [
    ...buildSessionRefs(sessions.ordered, this.#deps.sessionId),
    ...this.fileRefs,
  ]);
  // Prefix sources: any server provider that declared an autocomplete prefix. A
  // pick captures through that provider's key (carried on each result item).
  readonly refSources: RefSource[] = $derived.by(() =>
    this.serverProviders
      .filter((p) => p.autocompletePrefix)
      .map((p) => ({
        prefix: p.autocompletePrefix as string,
        label: p.label,
        search: async (q: string): Promise<RefItem[]> => {
          const sid = this.#deps.sessionId;
          if (!sid) return [];
          const results = await searchServerProvider(p.key, sid, q);
          return results.map((r) => ({
            id: r.value,
            kind: 'plugin' as const,
            label: r.label,
            providerKey: p.key,
            group: p.label,
          }));
        },
      })),
  );
  // The "add context" menu: the static client registry first, then any server
  // providers whose key doesn't collide with a client one (client wins, keeping
  // the keyed list unique). Client picks capture in the browser; server picks on
  // the daemon.
  readonly contextMenu: ContextMenuItem[] = $derived.by(() => [
    ...contextProviders.map((p) => ({
      key: p.key,
      label: p.label,
      icon: p.icon,
      kind: 'client' as const,
      hasChoices: false,
    })),
    ...this.serverProviders
      .filter((sp) => sp.inMenu && !contextProviders.some((cp) => cp.key === sp.key))
      .map((sp) => ({
        key: sp.key,
        label: sp.label,
        icon: sp.icon,
        kind: 'server' as const,
        hasChoices: sp.hasChoices,
        picker: sp.picker,
      })),
  ]);
  // Chips carry their provider's icon for the composer's attach row.
  readonly contextChips = $derived.by(() =>
    this.contextItems.map((it) => ({ ...it, icon: contextProvider(it.key)?.icon })),
  );

  constructor(deps: ContextDeps) {
    this.#deps = deps;
  }

  /** Load the daemon menu providers once (best-effort - empty if the daemon lacks
   *  the feature). */
  loadServerProviders(): void {
    fetchServerProviders().then((ps) => (this.serverProviders = ps));
  }

  /** Load the workspace files feeding the @ popover, once (best-effort). */
  loadFileRefs(): void {
    if (this.#refsLoaded) return;
    this.#refsLoaded = true;
    this.fileRefs = [];
    loadWorkspace()
      .then((ws) => {
        this.fileRefs = ws.entries
          .filter((e) => !e.is_dir)
          .map((e) => ({ id: e.path, kind: 'file' as const, label: e.path, group: 'Files' }));
      })
      .catch(() => {});
  }

  // Gather the context_metadata for an outgoing message: the manually-attached
  // chips, plus any provider whose auto-attach setting is on (best-effort
  // captured, a manual chip for the same provider winning). A capture failure
  // never blocks the send - it warns at most once per provider per session and
  // the message goes without that item.
  async resolveContextMetadata(): Promise<ContextItem[]> {
    const items = [...this.contextItems];
    for (const p of contextProviders) {
      if (!p.autoAttachStoreKey) continue;
      if (items.some((i) => i.key === p.key)) continue;
      if (!autoAttachStore(p.autoAttachStoreKey).enabled) continue;
      const res = await p.capture();
      if ('value' in res) {
        items.push({ key: p.key, label: p.label, value: res.value });
      } else if (!autoAttachWarned.has(p.key)) {
        autoAttachWarned.add(p.key);
        toasts.push('warn', `${p.label} not attached`, { body: res.error.message });
      }
    }
    return items;
  }

  // Add (or refresh) captured chips, replacing any existing item that shares an
  // incoming key so a re-pick updates in place rather than duplicating.
  addContextItems(incoming: ContextItem[]): void {
    if (!incoming.length) return;
    const keys = new Set(incoming.map((i) => i.key));
    this.contextItems = [...this.contextItems.filter((i) => !keys.has(i.key)), ...incoming];
  }

  // Attach a record named by a pasted reference marker to this composer: capture
  // it on the daemon and chip it. An empty capture (stale/unknown id) toasts; an
  // error toasts. Shares addContextItems with the manual-add and add-to-chat paths.
  async attachRef(kind: string, id: string): Promise<void> {
    const sid = this.#deps.sessionId;
    if (!sid) return;
    try {
      const items = await captureServerContext(kind, sid, id);
      if (items.length) this.addContextItems(items);
      else toasts.push('warn', 'Nothing to attach');
    } catch (err) {
      toasts.push('err', 'Could not attach', {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // Capture a daemon-side context item and chip it; a provider error toasts.
  // `arg` is a chosen submenu/picker value, or null for a no-choices provider.
  async captureServerItem(key: string, label: string, arg: string | null): Promise<void> {
    const sid = this.#deps.sessionId;
    if (!sid) return;
    try {
      this.addContextItems(await captureServerContext(key, sid, arg));
    } catch (err) {
      toasts.push('err', `${label} unavailable`, {
        body: err instanceof Error ? err.message : String(err),
      });
    }
  }

  // Pick a provider from the "add context" menu. A client provider captures in the
  // browser; a server provider captures on the daemon (with `arg` = a chosen
  // submenu value, or null). A picker provider (large, searchable option set)
  // opens the Picker overlay instead. Captured items become chips; a failure toasts.
  pickContext = async (item: ContextMenuItem, arg: string | null = null): Promise<void> => {
    if (item.kind === 'server') {
      if (!this.#deps.sessionId) return;
      if (item.picker && arg === null) {
        await this.openPicker(item);
        return;
      }
      await this.captureServerItem(item.key, item.label, arg);
      return;
    }
    const p = contextProvider(item.key);
    if (!p) return;
    const res = await p.capture();
    if ('value' in res) {
      this.addContextItems([{ key: p.key, label: p.label, value: res.value }]);
    } else {
      toasts.push(res.error.code === 'permission' ? 'warn' : 'err', `${p.label} unavailable`, {
        body: res.error.message,
      });
    }
  };

  // Open the Picker for a large-option provider: fetch its choices and show them
  // in a searchable overlay. Nothing to pick (empty or a failed load) simply
  // doesn't open, matching the inline-submenu behavior.
  async openPicker(item: ContextMenuItem): Promise<void> {
    const sid = this.#deps.sessionId;
    if (!sid) return;
    const choices = await fetchServerChoices(item.key, sid);
    if (choices.length === 0) return;
    this.picker = { item, items: choices };
  }

  pickFromPicker = (value: string): void => {
    const item = this.picker?.item;
    this.picker = null;
    if (item) void this.captureServerItem(item.key, item.label, value);
  };

  // A reference chosen in the @ popover attaches as a context chip. A file goes
  // through the workspace-file provider by its path (same as picking "Workspace
  // file" from the menu, and silent on an empty capture). A session or plugin
  // result attaches through its provider key (session's key IS `session`; a
  // plugin result carries its source provider key) with the ref id as the arg.
  pickRef = (ref: RefItem): void => {
    if (ref.kind === 'file') {
      void this.captureServerItem('file', ref.label, ref.label);
    } else {
      void this.attachRef(ref.providerKey ?? ref.kind, ref.id);
    }
  };

  // Load a server provider's submenu options, scoped to the active session.
  requestChoices = (key: string): Promise<ContextChoice[]> => {
    const sid = this.#deps.sessionId;
    return sid ? fetchServerChoices(key, sid) : Promise.resolve([]);
  };

  removeContext = (key: string): void => {
    this.contextItems = this.contextItems.filter((i) => i.key !== key);
  };
}
