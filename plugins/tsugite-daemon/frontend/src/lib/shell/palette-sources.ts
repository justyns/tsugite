/**
 * Command-palette sources the app shell owns: jump to a view, switch space, set
 * theme, open settings. Each row carries a `scheme:arg` href so selection is a
 * pure dispatch (`runPaletteHref`) with no parallel lookup table or object
 * identity to track. Job / pty / file sources plug in the same way - append their
 * rows to `buildPaletteItems` and teach `runPaletteHref` their scheme; chat
 * sessions use `buildSessionItems` instead, a query-only pool the palette folds
 * in under its own header (they must not crowd the default list).
 */
import type { IconName } from '$lib/components/icon/icons';
import type { PaletteItem } from '$lib/components/palette/palette-match';
import type { Theme } from '$lib/stores/theme.svelte';

/** A command param carrying the optional rich-input hints `/api/commands` serves:
 *  `widget` names a dedicated control (`model`, `effort`); `choices` is a fixed
 *  value set. Both drive argument autocomplete (the composer's inline choices, the
 *  palette's open-the-picker branch). */
export interface CommandParamLike {
  name: string;
  widget?: string;
  choices?: string[];
}

/** A slash command flattened to the fields the palette needs (name, description,
 *  and param hints), mirroring the shape `GET /api/commands` returns without
 *  coupling this module to the composer's fuller Command type. */
export interface CommandLike {
  name: string;
  description: string;
  params: CommandParamLike[];
}

export interface PaletteData {
  views: { id: string; label: string; icon: IconName }[];
  /** Plugin-contributed UI surfaces, openable as a mux tab. */
  surfaces: { kind: string; label: string; icon: IconName }[];
  themes: readonly Theme[];
  currentTheme: Theme;
  spaces: { id: string; name: string }[];
  activeSpaceId: string;
  commands: CommandLike[];
}

export interface PaletteContext {
  openView: (id: string) => void;
  openSurface: (kind: string) => void;
  setTheme: (theme: Theme) => void;
  setSpace: (id: string) => void;
  openSettings: () => void;
  openSession: (id: string) => void;
  newChat: () => void;
  showHelp: () => void;
  runCommand: (name: string) => void;
}

/** Params the composer auto-injects from context (never typed by the user). A
 *  command whose params are all auto-injected runs straight from the palette; any
 *  other param means the user must supply something, so we prefill instead. */
const AUTO_INJECTED_PARAMS = new Set(['user_id', 'session_id']);

/** Free-text params a bare `/command rest` maps its remainder onto, by convention. */
const PRIMARY_TEXT_PARAMS = ['prompt', 'message', 'cmd', 'task', 'query'];

/** Whether picking this command from the palette should prefill the composer (the
 *  user still has an argument to type) rather than run it outright. */
export function commandNeedsInput(cmd: { params: { name: string }[] }): boolean {
  return !cmd.params.every((p) => AUTO_INJECTED_PARAMS.has(p.name));
}

/** The single param a `/command arg` maps its argument onto: a known free-text
 *  param if present, else the first param the user must supply. This is the param
 *  whose `widget`/`choices` hint drives argument autocomplete. Generic so callers
 *  keep their own param type (and read its widget/choices off the result). */
export function commandArgParam<P extends { name: string }>(cmd: { params: P[] }): P | undefined {
  const named = PRIMARY_TEXT_PARAMS.find((n) => cmd.params.some((p) => p.name === n));
  if (named) return cmd.params.find((p) => p.name === named);
  return cmd.params.find((p) => !AUTO_INJECTED_PARAMS.has(p.name));
}

/** What selecting a command from the ⌘K palette should do, decided from its arg
 *  param's hint: a `widget:"model"` arg opens the header model picker; anything
 *  else runs outright (no user input) or prefills `/name ` so the composer's own
 *  slash flow (inline choices for effort/choices params, or a text field) takes
 *  over. The caller owns navigation and dispatching the chosen action. */
export type CommandAction =
  { kind: 'model-picker' } | { kind: 'run'; text: string } | { kind: 'prefill'; text: string };

export function commandPaletteAction(cmd: {
  name: string;
  params: CommandParamLike[];
}): CommandAction {
  if (commandArgParam(cmd)?.widget === 'model') return { kind: 'model-picker' };
  return commandNeedsInput(cmd)
    ? { kind: 'prefill', text: `/${cmd.name} ` }
    : { kind: 'run', text: `/${cmd.name}` };
}

/** A chat session flattened to the fields the palette needs, so this module
 *  stays decoupled from the sessions store row. The caller resolves the display
 *  title, liveness, age stamp, and topic (see App.svelte). */
export interface SessionLike {
  id: string;
  title: string;
  ended: boolean;
  when: string;
  topic: string;
  /** A pre-compaction session whose successor carries the thread on. Excluded
   *  from the pool so a compacted chat surfaces once, not as a title-twin pair. */
  superseded?: boolean;
}

export function buildPaletteItems(data: PaletteData): PaletteItem[] {
  const items: PaletteItem[] = [];

  for (const view of data.views) {
    items.push({
      group: 'views',
      icon: view.icon,
      label: view.label,
      meta: 'view',
      href: `view:${view.id}`,
    });
  }

  for (const surface of data.surfaces) {
    items.push({
      group: 'plugins',
      icon: surface.icon,
      label: surface.label,
      meta: 'tab',
      href: `surface:${surface.kind}`,
    });
  }

  for (const space of data.spaces) {
    items.push({
      group: 'spaces',
      icon: 'grid',
      label: space.name,
      meta: space.id === data.activeSpaceId ? 'current' : 'space',
      href: `space:${space.id}`,
    });
  }

  for (const theme of data.themes) {
    items.push({
      group: 'theme',
      icon: 'sparkle',
      label: theme,
      meta: theme === data.currentTheme ? 'current' : 'theme',
      href: `theme:${theme}`,
    });
  }

  items.push({
    group: 'actions',
    icon: 'plus',
    label: 'New chat',
    meta: 'action',
    href: 'action:new-chat',
    keywords: 'create session new conversation',
  });

  items.push({
    group: 'actions',
    icon: 'q',
    label: 'Keyboard shortcuts',
    meta: 'action',
    href: 'action:shortcuts',
    keywords: 'keys hotkeys bindings help',
  });

  items.push({
    group: 'actions',
    icon: 'tool',
    label: 'Settings',
    meta: 'action',
    href: 'action:settings',
  });

  // Slash commands as an additional ⌘K entry point (the composer's inline `/`
  // menu is unchanged). The description rides `keywords` so a query matches on
  // what a command does, not just its name.
  for (const cmd of data.commands) {
    items.push({
      group: 'commands',
      icon: 'term',
      label: `/${cmd.name}`,
      meta: cmd.description || 'command',
      href: `command:${cmd.name}`,
      keywords: cmd.description,
    });
  }

  return items;
}

/** Map chat sessions to palette rows, live sessions before ended ones (recency
 *  preserved within each). Fed to the palette as a query-only pool, so they never
 *  crowd the default list but a matching title/topic surfaces them. */
export function buildSessionItems(sessions: SessionLike[]): PaletteItem[] {
  const live: PaletteItem[] = [];
  const ended: PaletteItem[] = [];
  for (const s of sessions) {
    if (s.superseded) continue;
    const item: PaletteItem = {
      group: 'sessions',
      icon: 'chat',
      label: s.title,
      meta: s.when,
      href: `session:${s.id}`,
      keywords: s.topic,
    };
    (s.ended ? ended : live).push(item);
  }
  return [...live, ...ended];
}

/** Run a palette row's command href. Returns whether the scheme was recognised. */
export function runPaletteHref(href: string | undefined, ctx: PaletteContext): boolean {
  if (!href) return false;
  const colon = href.indexOf(':');
  if (colon === -1) return false;
  const scheme = href.slice(0, colon);
  const arg = href.slice(colon + 1);
  switch (scheme) {
    case 'view':
      ctx.openView(arg);
      return true;
    case 'theme':
      ctx.setTheme(arg as Theme);
      return true;
    case 'space':
      ctx.setSpace(arg);
      return true;
    case 'surface':
      ctx.openSurface(arg);
      return true;
    case 'session':
      ctx.openSession(arg);
      return true;
    case 'command':
      ctx.runCommand(arg);
      return true;
    case 'action':
      if (arg === 'new-chat') {
        ctx.newChat();
        return true;
      }
      if (arg === 'settings') {
        ctx.openSettings();
        return true;
      }
      if (arg === 'shortcuts') {
        ctx.showHelp();
        return true;
      }
      return false;
    default:
      return false;
  }
}
