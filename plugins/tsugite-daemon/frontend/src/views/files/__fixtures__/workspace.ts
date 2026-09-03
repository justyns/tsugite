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

const INITIAL_CONTENT: Record<string, string> = {
  'index.md': '# Home\n\ntags: #home\n\nStart at [[alpha]].\n',
  'ops/alpha.md':
    '# Alpha\n\ntags: #ops #x\n\nSee [[beta]] and [[ghost]].\n\n## Section\n\nselectable paragraph\n',
  'ops/beta.md': '# Beta\n\ntags: #ops #x\n\nBack to [[alpha]] for context.\n',
  // A generated report: a relative stylesheet, an external one it must not
  // fetch, and a script that must never run.
  'reports/report.html':
    '<html><head><link rel="stylesheet" href="report.css">' +
    '<link rel="stylesheet" href="https://cdn.example.com/evil.css"></head>' +
    '<body><h1>Coverage</h1><script>parent.steal()</scr' +
    'ipt></body></html>',
  'reports/report.css': 'h1 { color: rebeccapurple }',
};

let CONTENT: Record<string, string> = { ...INITIAL_CONTENT };

const DIRS: Record<string, Entry[]> = {
  '': [
    { path: 'ops', name: 'ops', is_dir: true },
    { path: 'reports', name: 'reports', is_dir: true },
    {
      path: 'index.md',
      name: 'index.md',
      is_dir: false,
      size: 40,
      modified: '2026-07-14T00:00:00Z',
    },
  ],
  reports: [
    {
      path: 'reports/report.html',
      name: 'report.html',
      is_dir: false,
      size: 200,
      modified: '2026-07-14T00:00:00Z',
    },
    {
      path: 'reports/report.css',
      name: 'report.css',
      is_dir: false,
      size: 30,
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
  /** Stand in for an agent's tool rewriting a file underneath an open tab. */
  setContent: (path: string, content: string) => {
    CONTENT[path] = content;
  },
  reset: () => {
    CONTENT = { ...INITIAL_CONTENT };
  },
  api: {
    get: async (path: string) => {
      if (path === '/api/runtime') {
        return {
          agent_file: 'smoke',
          workspace_dir: '/ws/smoke',
          model: null,
          context_limit: null,
          running_tasks: 0,
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
