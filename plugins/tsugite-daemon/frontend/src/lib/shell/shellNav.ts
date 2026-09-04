/**
 * Bridge between the multiplexer's surface vocabulary and the nav rail's view
 * vocabulary. A docked *surface* is named for what it shows ('chat', 'terminal',
 * 'file'); the nav rail names the *view* that lists them ('chats', 'terminals',
 * 'files'). This map is the single source of truth, so the rail highlight follows
 * whatever surface is focused in the mux.
 */
import { type Layout, type PaneTabModel, collectLeaves, findLeaf } from './mux/layout';

// Only the surfaces that also have a shorter docking name need an alias; every
// other view id doubles as its own surface kind.
const VIEW_FOR_SURFACE: Record<string, string> = {
  chat: 'chats',
  terminal: 'terminals',
  file: 'files',
};

/** The nav view id a docked surface belongs to (drives the rail highlight). */
export function surfaceViewId(kind: string): string {
  return VIEW_FOR_SURFACE[kind] ?? kind;
}

/** The active tab (surface) of the layout's focused pane, or null when the focused
 *  pane is empty. Used by the shell to point the context rail's highlight at (and
 *  read the params of) whatever surface currently has focus. */
export function focusedSurface(layout: Layout): PaneTabModel | null {
  const leaf =
    (layout.focusedPaneId && findLeaf(layout, layout.focusedPaneId)) ||
    collectLeaves(layout.root)[0];
  if (!leaf) return null;
  return leaf.tabs.find((t) => t.id === leaf.activeTabId) ?? null;
}

/**
 * The view id of the surface active in the layout's focused pane. Empty panes or
 * a dangling focus id resolve to the first leaf, then to '' - the caller supplies
 * the default view.
 */
export function focusedViewId(layout: Layout): string {
  const tab = focusedSurface(layout);
  return tab ? surfaceViewId(tab.kind) : '';
}

/**
 * The sessionId of the chat currently on screen: the focused pane's chat tab if
 * focus is on one, else any chat tab docked in the layout (normally just the
 * one, kept in place by retargetOrOpen). Null when no chat is docked.
 *
 * Deliberately independent of pane focus for panes that aren't chats - clicking
 * into (say) an artifact pane beside the chat must not read as "no chat is open"
 * for callers scoping that pane to a conversation.
 */
export function dockedChatSessionId(layout: Layout): string | null {
  const focused = focusedSurface(layout);
  if (focused?.kind === 'chat') return focused.params.sessionId ?? null;
  for (const leaf of collectLeaves(layout.root)) {
    const tab = leaf.tabs.find((t) => t.kind === 'chat');
    if (tab) return tab.params.sessionId ?? null;
  }
  return null;
}
