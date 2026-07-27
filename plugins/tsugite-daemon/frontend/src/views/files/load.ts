/**
 * Loads an agent's workspace for the client-side wiki. `loadWorkspace` asks the
 * workspace endpoint to walk the tree server-side and return the whole flat
 * listing in ONE request (listings only - it reads no file contents), then ships
 * a paths-only index, so opening the Files view costs a single directory GET
 * instead of one round-trip per directory. The content-derived backlink/tag index
 * is a separate, explicitly-requested pass (`loadContentIndex`) because there is
 * no server-side index and building it client-side means reading every markdown
 * file once.
 */
import { api } from '$lib/api/client';
import { files, type WorkspaceEntry } from '$lib/stores/files.svelte';
import { buildIndex, buildPathsIndex, buildTree, type TreeNode, type WikiIndex } from './wiki';

const MARKDOWN = /\.(md|markdown)$/i;
/** Cap the number of files whose contents feed the backlink index. The tree
 *  still lists everything; only the link/tag graph is bounded. */
const MAX_INDEXED = 400;

export interface Workspace {
  entries: WorkspaceEntry[];
  tree: TreeNode[];
  index: WikiIndex;
  workspaceDir: string;
  /** True when the server capped the recursive listing, so the tree is partial. */
  truncated?: boolean;
}

export function isMarkdown(name: string): boolean {
  return MARKDOWN.test(name);
}

function markdownPaths(entries: WorkspaceEntry[]): string[] {
  return entries.filter((e) => !e.is_dir && isMarkdown(e.name)).map((e) => e.path);
}

// The files store exposes only a stateful single-slot `list()`; loading the wiki
// needs a stateless read of the whole tree, so this hits the workspace endpoint
// directly. `recursive` asks the daemon to walk the tree server-side and return
// the full flat listing in one response (it sets `truncated` if it hit its cap).
async function readDir(
  agent: string,
  subdir: string,
  opts: { recursive?: boolean } = {},
): Promise<{ entries: WorkspaceEntry[]; workspaceDir: string; truncated: boolean }> {
  const params = new URLSearchParams();
  if (subdir) params.set('subdir', subdir);
  if (opts.recursive) params.set('recursive', '1');
  const qs = params.toString();
  const res = await api.get<{
    entries: WorkspaceEntry[];
    workspace_dir: string;
    truncated?: boolean;
  }>(`/api/agents/${encodeURIComponent(agent)}/workspace${qs ? `?${qs}` : ''}`);
  return {
    entries: res.entries,
    workspaceDir: res.workspace_dir,
    truncated: res.truncated ?? false,
  };
}

export async function loadWorkspace(agent: string): Promise<Workspace> {
  const { entries, workspaceDir, truncated } = await readDir(agent, '', { recursive: true });
  return {
    entries,
    tree: buildTree(entries),
    index: buildPathsIndex(markdownPaths(entries)),
    workspaceDir,
    truncated,
  };
}

/** Read every markdown file once (capped) and build the full backlink/tag
 *  index. On-demand only - this is the whole-workspace download. */
export async function loadContentIndex(
  agent: string,
  entries: WorkspaceEntry[],
): Promise<WikiIndex> {
  const paths = markdownPaths(entries).slice(0, MAX_INDEXED);
  const docs = await Promise.all(
    paths.map(async (path) => {
      try {
        const file = await files.read(agent, path);
        return { path, content: file.content ?? '' };
      } catch {
        return { path, content: '' };
      }
    }),
  );
  return buildIndex(docs);
}
