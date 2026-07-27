import type { Route, RouteParams } from '$lib/router.svelte';
import { navigate, routeHistory } from '$lib/router.svelte';
import type { WorkspaceView } from '$lib/stores/shellView.svelte';

// Phone drilldown, shared by all three workspace views (chats/terminals/files). At
// phone width each view is two first-class screens - the rail/list, and one item's
// content - switched by the hash, not a toggled drawer: a rail pick sets the view's
// content param, the back affordance clears it. Desktop/tablet render both side by
// side and never consult this.

// The hash param that, when present, means a view is showing an ITEM (its content
// screen). Each workspace rail navigates with its own key.
export const CONTENT_PARAM: Record<WorkspaceView, string> = {
  chats: 'sessionId',
  terminals: 'terminalId',
  files: 'path',
};

export type PhoneScreen = 'list' | 'content';

/** The app's phone breakpoint as a point-in-time check (the same `<=640px` media
 *  query the shell reads reactively). Use for one-shot decisions at an event -
 *  e.g. whether landing on a chat should auto-focus the composer - where a
 *  reactive `narrow` flag isn't wanted (a later resize must not re-trigger). */
export function isPhoneWidth(): boolean {
  return typeof window !== 'undefined' && !!window.matchMedia?.('(max-width: 640px)').matches;
}

export function workspacePhoneScreen(args: {
  /** Phone width (the app's <=640px breakpoint). */
  narrow: boolean;
  /** The hash router's current view. */
  view: string;
  /** The active workspace context rail. */
  workspaceView: WorkspaceView;
  /** The hash's params. */
  params: RouteParams;
}): PhoneScreen | null {
  if (!args.narrow) return null;
  // The content param only counts when the hash actually reads this workspace view:
  // an empty boot hash or a full-view hash (e.g. #jobs) over the restored workspace
  // is still the list.
  const hasContent =
    args.view === args.workspaceView && !!args.params[CONTENT_PARAM[args.workspaceView]];
  return hasContent ? 'content' : 'list';
}

/**
 * The back affordance's target from a phone content screen. Popping when the list
 * is the entry behind us keeps the history stack clean and matches a hardware back
 * in one tap; a cold entry (deep link, PWA restore, cross-view jump) has no list
 * behind it, so we push the bare view hash rather than escape the app.
 */
export function workspaceBackAction(
  prev: Route | null,
  view: WorkspaceView,
): { kind: 'pop' } | { kind: 'list' } {
  if (prev && prev.view === view && !prev.params[CONTENT_PARAM[view]]) return { kind: 'pop' };
  return { kind: 'list' };
}

/** Imperative back for a phone workspace drilldown; each surface's back control
 *  calls this with its own view. */
export function goBackToWorkspaceList(view: WorkspaceView): void {
  const action = workspaceBackAction(routeHistory.prev, view);
  if (action.kind === 'pop') history.back();
  else navigate(view);
}
