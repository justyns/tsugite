import type { IconName } from '$lib/components/icon/icons';

/** Kind of entity a reference points at - selects the row's leading glyph.
 *  `session` and `plugin` attach as a context chip (like `file`); the rest
 *  insert their label inline. */
export type RefKind = 'file' | 'chat' | 'agent' | 'terminal' | 'session' | 'plugin';

/** Git working-tree status for a file reference (letter + color + title). */
export type GitStatus = 'm' | 'a' | 'u' | 'd';

/** A single suggestion in the RefAutocomplete popover. */
export interface RefItem {
  id: string;
  kind: RefKind;
  /** Display text, e.g. `@sse-reconnect.md` (trigger included). */
  label: string;
  /** Muted trailing detail, e.g. `kb/ops · modified`. */
  detail?: string;
  /** Optional git file-state glyph for file references. */
  git?: GitStatus;
  /** The context-provider key a pick captures under. Defaults to `kind` (a
   *  `session` item captures via the `session` provider); a `plugin` item carries
   *  its source provider's key here since its kind is the generic `plugin`. */
  providerKey?: string;
  /** Section label for grouped results (`Sessions`, `Files`, a source's label). */
  group?: string;
}

/** A prefix-scoped, query-aware `@` autocomplete source. Typing `@<prefix> <query>`
 *  switches the popover to this source and (debounced) calls `search(query)`; its
 *  results are shown as-is, already server-filtered. */
export interface RefSource {
  /** The bare `@` prefix that activates this source, e.g. `jira`. */
  prefix: string;
  /** Section label shown above this source's results. */
  label: string;
  search: (query: string) => Promise<RefItem[]>;
}

/** A staged attachment chip shown in the composer's attach row. */
export interface Attachment {
  id: string;
  name: string;
  /** Pre-formatted size label, e.g. `118 KB`. */
  size?: string;
}

/** An attached client-context item, shown as a removable chip in the attach row
 *  and sent as message metadata. `icon` names the provider's glyph. */
export interface ContextChip {
  key: string;
  label: string;
  value: string;
  icon?: IconName;
}

/** One provider offered in the composer's "add context" menu. */
export interface ContextMenuItem {
  key: string;
  label: string;
  icon: IconName;
  /** Where a pick is captured: in the browser (client) or on the daemon (server).
   *  Absent is treated as a client provider. */
  kind?: 'client' | 'server';
  /** A server provider that offers a submenu of choices before it captures. */
  hasChoices?: boolean;
  /** A server provider whose option set is large/searchable: picking it opens the
   *  Picker overlay instead of the inline submenu. */
  picker?: boolean;
}

/** One option in a server provider's submenu; its `value` is sent to capture as
 *  the `arg`, its `label` names the row. */
export interface ContextChoice {
  value: string;
  label: string;
}
