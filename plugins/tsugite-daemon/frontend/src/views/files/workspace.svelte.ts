/**
 * Shared workspace model for the files rail + file surfaces. `loadWorkspace`
 * walks directory listings only (cheap), so the rail tree and wikilink
 * resolution come up without downloading any file contents. The backlink/tag
 * index needs every markdown file read once, so it stays lazy: `ensureIndex()`
 * runs that scan the first time something actually asks for it (the meta
 * pane's scan affordance, a #tag search) and re-runs it after saves once
 * built. Exported as a class instance - never a reassigned binding.
 */
import { loadContentIndex, loadWorkspace, type Workspace } from './load';

export type IndexState = 'none' | 'building' | 'ready';

export class FilesWorkspaceStore {
  agent = $state('');
  ws = $state<Workspace | null>(null);
  loading = $state(false);
  error = $state<string | null>(null);
  /** Lifecycle of the content-derived backlink/tag index. */
  indexState = $state<IndexState>('none');
  private inflight: Promise<void> | null = null;
  // Keyed by the workspace object it scans, so a reload mid-scan starts a fresh
  // scan for the new tree instead of adopting (or blocking on) the stale one.
  private indexInflight: { ws: Workspace; run: Promise<void> } | null = null;

  /** Load the agent's workspace if it isn't already loaded (or loading). */
  async ensure(agent: string): Promise<void> {
    if (agent === this.agent && (this.ws || this.inflight)) {
      await this.inflight;
      return;
    }
    await this.reload(agent);
  }

  /** Force a fresh walk (after a save changes the tree or link graph). A
   *  content index that was already built or requested is rebuilt for the new
   *  tree; one never requested stays unbuilt. */
  async reload(agent: string): Promise<void> {
    const rebuildIndex = this.indexState !== 'none' && agent === this.agent;
    this.agent = agent;
    this.loading = true;
    this.error = null;
    this.indexState = 'none';
    const run = (async () => {
      try {
        this.ws = await loadWorkspace(agent);
        if (rebuildIndex) void this.ensureIndex();
      } catch (err) {
        this.error = err instanceof Error ? err.message : String(err);
      } finally {
        this.loading = false;
        this.inflight = null;
      }
    })();
    this.inflight = run;
    await run;
  }

  /** Build the full backlink/tag index on demand (reads every markdown file
   *  once). Safe to call repeatedly; concurrent calls share one scan. */
  async ensureIndex(): Promise<void> {
    const ws = this.ws;
    if (!ws || this.indexState === 'ready') return;
    if (this.indexInflight?.ws === ws) return this.indexInflight.run;
    this.indexState = 'building';
    const run = (async () => {
      try {
        const index = await loadContentIndex(this.agent, ws.entries);
        // A reload may have swapped the workspace mid-scan; only publish onto
        // the workspace the scan read from.
        if (this.ws === ws) {
          this.ws = { ...ws, index };
          this.indexState = 'ready';
        }
      } catch {
        if (this.ws === ws) this.indexState = 'none';
      } finally {
        if (this.indexInflight?.ws === ws) this.indexInflight = null;
      }
    })();
    this.indexInflight = { ws, run };
    return run;
  }
}

export const filesWorkspace = new FilesWorkspaceStore();
