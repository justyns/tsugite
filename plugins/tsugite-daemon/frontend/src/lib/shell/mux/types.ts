/**
 * Shared prop contracts for the mux component tree. The handler bag is threaded
 * unchanged from `Mux` down through the recursive `MuxNode` to each `PaneView`,
 * so a single spread forwards every callback instead of re-listing them.
 */
import type { Snippet } from 'svelte';
import type { PaneTabModel, SplitDir, SurfaceRef } from './layout';

export interface MuxHandlers {
  onSelectTab?: (paneId: string, tabId: string) => void;
  /** Pin a preview (ephemeral) tab so the next preview opens fresh instead of
   *  replacing it; fired by a double-click on the tab. */
  onPinTab?: (paneId: string, tabId: string) => void;
  onCloseTab?: (paneId: string, tabId: string) => void;
  onCloseOtherTabs?: (paneId: string, tabId: string) => void;
  onCloseAllTabs?: (paneId: string) => void;
  /** New-tab (+) affordance; when omitted the button is hidden. Wired by the
   *  chrome to open the command palette for the target pane. */
  onNewTab?: (paneId: string) => void;
  onFocusPane?: (paneId: string) => void;
  // Arg order mirrors the `splitPane` reducer / store `split` method so the
  // chrome can forward it directly.
  onSplit?: (paneId: string, dir: SplitDir, ref: SurfaceRef, position: 'before' | 'after') => void;
  onDock?: (paneId: string, ref: SurfaceRef) => void;
  onResize?: (splitId: string, dividerIndex: number, deltaFraction: number) => void;
}

/** Renders a docked surface by its {kind, params}; supplied by the view host. */
export type MuxContent = Snippet<[PaneTabModel]>;
