/**
 * In-memory workspace fixture shared by the files rail + surface browser tests.
 * The store + loader talk to this through a mocked api client, so the tree, the
 * wiki index, and the rendered doc all run for real in chromium without a daemon.
 */
interface Entry {
  path: string;
  name: string;
  is_dir: boolean;
  size?: number;
  modified?: string;
}

const CONTENT: Record<string, string> = {
  'index.md': '# Home\n\ntags: #home\n\nStart at [[alpha]].\n',
  'ops/alpha.md': '# Alpha\n\ntags: #ops #x\n\nSee [[beta]] and [[ghost]].\n\n## Section\n\nbody\n',
  'ops/beta.md': '# Beta\n\ntags: #ops #x\n\nBack to [[alpha]] for context.\n',
};

const DIRS: Record<string, Entry[]> = {
  '': [
    { path: 'ops', name: 'ops', is_dir: true },
    {
      path: 'index.md',
      name: 'index.md',
      is_dir: false,
      size: 40,
      modified: '2026-07-14T00:00:00Z',
    },
  ],
  ops: [
    {
      path: 'ops/alpha.md',
      name: 'alpha.md',
      is_dir: false,
      size: 60,
      modified: '2026-07-14T00:00:00Z',
    },
    {
      path: 'ops/beta.md',
      name: 'beta.md',
      is_dir: false,
      size: 40,
      modified: '2026-07-14T00:00:00Z',
    },
  ],
};

export const WORKSPACE = {
  api: {
    get: async (path: string) => {
      if (path === '/api/agents') {
        return {
          agents: [
            { name: 'smoke', agent_file: 'smoke.md', workspace_dir: '/ws/smoke', running_tasks: 0 },
          ],
        };
      }
      const url = new URL(path, 'http://x');
      if (url.pathname.endsWith('/workspace/content')) {
        const p = url.searchParams.get('path') ?? '';
        return { path: p, content: CONTENT[p] ?? '', is_text: true };
      }
      if (url.pathname.endsWith('/workspace')) {
        if (url.searchParams.get('recursive')) {
          // The daemon walks the tree server-side and returns the whole flat
          // listing in one response; loadWorkspace uses this recursive form.
          return { entries: Object.values(DIRS).flat(), workspace_dir: '/ws/smoke' };
        }
        const subdir = url.searchParams.get('subdir') ?? '';
        return { entries: DIRS[subdir] ?? [], subdir, workspace_dir: '/ws/smoke' };
      }
      throw new Error(`unexpected GET ${path}`);
    },
    put: async () => ({ status: 'saved' }),
    post: async () => ({
      files: [
        {
          name: 'alpha.md',
          content_type: 'text/markdown',
          mime_type: 'text/markdown',
          size: 60,
          context_attach: true,
        },
      ],
    }),
  },
};
